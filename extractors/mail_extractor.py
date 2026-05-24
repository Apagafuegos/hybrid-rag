"""
Concrete Mailing List Extractor
================================
Parses standard ``.mbox`` archives using Python's native ``mailbox``
module and maps each message body into a ``UnifiedChunk``.
"""

from __future__ import annotations

import email
import mailbox
import re
import uuid
from pathlib import Path
from typing import List, Optional

from core.models import UnifiedChunk, UnifiedChunkMetadata
from extractors.base import DataExtractor


class MailingListExtractor(DataExtractor):
    """
    Extractor for Linux Kernel Mailing List (LKML) ``.mbox`` archives.

    Cleans cryptographic signatures and empty header margins, then
    isolates conversational text blocks into ``UnifiedChunk`` segments.
    """

    SUPPORTED_EXTENSIONS = {".mbox", ".txt"}

    def extract_chunks(self, source_path: str) -> List[UnifiedChunk]:
        path = Path(source_path)
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"MailingListExtractor does not support '{path.suffix}'. "
                f"Expected one of {self.SUPPORTED_EXTENSIONS}."
            )

        # mailbox.mbox requires the file to exist on disk with proper mbox formatting
        chunks: List[UnifiedChunk] = []

        if path.suffix == ".mbox":
            mbox = mailbox.mbox(str(path))
            for key, msg in mbox.items():
                chunk = self._map_message(msg, source_path, key)
                if chunk is not None:
                    chunks.append(chunk)
        else:
            # Fallback: treat plain text as a single synthetic email
            text = path.read_text(encoding="utf-8", errors="replace")
            cleaned = self._clean_body(text)
            if cleaned.strip():
                chunks.append(
                    self._build_chunk(
                        body=cleaned,
                        msg_id=f"{path.name}#0",
                        subject="Synthetic email",
                        sender="unknown@localhost",
                        date="",
                        in_reply_to="",
                        references="",
                        file_path=source_path,
                    )
                )

        return chunks

    def _map_message(
        self, msg: email.message.Message, file_path: str, mbox_key: str
    ) -> Optional[UnifiedChunk]:
        body = self._extract_body(msg)
        if not body or not body.strip():
            return None

        cleaned = self._clean_body(body)
        if not cleaned.strip():
            return None

        msg_id = msg.get("Message-Id", f"{file_path}#{mbox_key}").strip()
        subject = msg.get("Subject", "").strip()
        sender = msg.get("From", "").strip()
        date = msg.get("Date", "").strip()
        in_reply_to = msg.get("In-Reply-To", "").strip()
        references = msg.get("References", "").strip()

        return self._build_chunk(
            body=cleaned,
            msg_id=msg_id,
            subject=subject,
            sender=sender,
            date=date,
            in_reply_to=in_reply_to,
            references=references,
            file_path=file_path,
        )

    def _build_chunk(
        self,
        *,
        body: str,
        msg_id: str,
        subject: str,
        sender: str,
        date: str,
        in_reply_to: str,
        references: str,
        file_path: str,
    ) -> UnifiedChunk:
        thread_id = in_reply_to or references or msg_id
        tokens = self._tokenize(body, subject, sender)

        return UnifiedChunk(
            id=str(uuid.uuid4()),
            text_content=body,
            source_type="lkml_email",
            source_id=msg_id,
            sparse_tokens={"tokens": tokens},
            metadata=UnifiedChunkMetadata(
                hierarchical_tags=["lkml", "email", "conversation"],
                parent_structure=subject or "(no subject)",
                file_path_or_url=file_path,
                custom_attributes={
                    "sender": sender,
                    "subject": subject,
                    "date": date,
                    "thread_id": thread_id,
                    "in_reply_to": in_reply_to,
                    "references": references,
                },
            ),
        )

    @staticmethod
    def _extract_body(msg: email.message.Message) -> str:
        """Walk the MIME tree and concatenate text/plain parts."""
        parts: List[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode("utf-8", errors="replace"))
        return "\n".join(parts)

    @staticmethod
    def _clean_body(text: str) -> str:
        """
        Strip PGP signatures, excessive blank lines, and typical
        mailing-list footer noise.
        """
        # Remove PGP signature blocks
        text = re.sub(
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",
            "",
            text,
            flags=re.DOTALL,
        )
        # Remove PGP public key blocks
        text = re.sub(
            r"-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----",
            "",
            text,
            flags=re.DOTALL,
        )
        # Normalize excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _tokenize(body: str, subject: str, sender: str) -> List[str]:
        """Extract keyword hints for sparse retrieval."""
        combined = f"{subject} {sender} {body}"
        words = re.findall(r"[A-Za-z]\w+", combined)
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "day", "get", "has",
            "him", "his", "how", "its", "may", "new", "now", "old", "see",
            "two", "who", "boy", "did", "she", "use", "her", "way", "many",
        }
        tokens = sorted({w.lower() for w in words if len(w) > 3 and w.lower() not in stopwords})
        return tokens[:64]

"""
Encryption/Decryption module for API keys
Compatible with TypeScript AES-256-GCM encryption
Matches the exact format: IV (16 bytes) + authTag (16 bytes) + encrypted data
"""

import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)

# Constants (matching TypeScript implementation)
IV_LENGTH = 16  # 16 bytes (128 bits)
AUTH_TAG_LENGTH = 16  # 16 bytes (128 bits)
ALGORITHM = "aes-256-gcm"


class EncryptionManager:
    """Handles encryption and decryption of API keys using AES-256-GCM"""

    def __init__(self, encryption_key: str = None):
        """
        Initialize encryption manager

        Args:
            encryption_key: The encryption key (from environment)
                          If not provided, uses API_KEY_ENCRYPTION_SECRET from environment
        """
        self.encryption_key = encryption_key or os.getenv('API_KEY_ENCRYPTION_SECRET')

        if not self.encryption_key:
            logger.warning("API_KEY_ENCRYPTION_SECRET not set, using default (NOT SECURE!)")
            self.encryption_key = "default-32-byte-encryption-key!!"

        # Pad/trim key to 32 bytes (256 bits) - matching TypeScript logic
        self.key = self._prepare_key(self.encryption_key)

    def _prepare_key(self, key: str) -> bytes:
        """
        Prepare encryption key - pad to 32 bytes or trim
        Matches TypeScript: ENCRYPTION_KEY.padEnd(32, "0").slice(0, 32)
        """
        # Pad with '0' to 32 chars if shorter, or trim to 32 if longer
        padded_key = key.ljust(32, '0')[:32]
        return padded_key.encode('utf-8')

    def encrypt(self, text: str) -> str:
        """
        Encrypt a string (e.g., API key) using AES-256-GCM
        Returns encrypted data as base64 string with IV and auth tag prepended

        Args:
            text: Plain text to encrypt

        Returns:
            Base64 encoded string containing: IV (16) + authTag (16) + encrypted data
        """
        if not text:
            return ""

        try:
            # Generate random IV (16 bytes)
            iv = os.urandom(IV_LENGTH)

            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Encrypt the text
            ciphertext = encryptor.update(text.encode('utf-8')) + encryptor.finalize()

            # Get the auth tag
            auth_tag = encryptor.tag

            # Combine IV + authTag + encrypted data (matching TypeScript)
            combined = iv + auth_tag + ciphertext

            # Return as base64
            return base64.b64encode(combined).decode('utf-8')

        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return ""

    def decrypt(self, encrypted_text: str) -> str:
        """
        Decrypt a string that was encrypted using the encrypt function
        Compatible with TypeScript encryption format

        Args:
            encrypted_text: Base64 encoded encrypted string

        Returns:
            Decrypted plain text string
        """
        if not encrypted_text:
            return ""

        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_text)

            # Extract IV, authTag, and encrypted data (matching TypeScript)
            iv = combined[:IV_LENGTH]
            auth_tag = combined[IV_LENGTH:IV_LENGTH + AUTH_TAG_LENGTH]
            ciphertext = combined[IV_LENGTH + AUTH_TAG_LENGTH:]

            # Create decipher
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            # Decrypt the data
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()

            return decrypted.decode('utf-8')

        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return ""


# Singleton instance
_encryption_manager = None


def get_encryption_manager() -> EncryptionManager:
    """
    Get or create singleton encryption manager instance

    Returns:
        EncryptionManager instance
    """
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Convenience function to decrypt an API key

    Args:
        encrypted_key: Encrypted API key string

    Returns:
        Decrypted API key string
    """
    if not encrypted_key:
        return ""

    encryption_manager = get_encryption_manager()
    return encryption_manager.decrypt(encrypted_key)

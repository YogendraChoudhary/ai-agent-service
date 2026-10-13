"""
Redis Manager for License Validation and User API Keys
Handles all Redis operations for the AI agent API
"""

import redis
import json
import os
import re
import ssl
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging
import certifi
from dateutil import parser as dateutil_parser
from .encryption import decrypt_api_key

logger = logging.getLogger(__name__)


def parse_expiry_date(date_str: Any) -> Optional[datetime]:
    """
    Parse an expiration date string into a datetime object.

    Handles both formats found in the Redis license database:
      1. ISO 8601 strings (e.g. '2026-12-18T11:17:57.387Z')
      2. JavaScript Date.toString() strings (e.g. 'Thu Dec 10 2026 21:52:20 GMT+0530 (India Standard Time)')
      3. Epoch timestamps (seconds)
    """
    if not date_str:
        return None
    date_str = str(date_str).strip()
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        pass

    try:
        # Strip parenthesized timezone label: (India Standard Time)
        cleaned = re.sub(r'\(.*?\)', '', date_str).strip()
        # Normalize GMT+0530 / GMT-0500 to +0530 / -0500 for dateutil
        cleaned = re.sub(r'GMT([+-]\d{4})', r'\1', cleaned)
        return dateutil_parser.parse(cleaned)
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(float(date_str), tz=timezone.utc)
    except (ValueError, TypeError):
        pass

    return None


class RedisManager:
    """Manages Redis operations for license validation and user API keys"""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis connection

        Args:
            redis_url: Redis connection URL (format: redis://username:password@host:port)
                      If not provided, uses REDIS_URL from environment
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL')

        if not self.redis_url:
            raise ValueError("REDIS_URL environment variable is required")

        # Parse Redis URL and connect with robust settings
        try:
            # For rediss:// (SSL) URLs, provide CA certs so verification works on macOS
            ssl_kwargs = {}
            if self.redis_url.startswith('rediss://'):
                ssl_kwargs['ssl_ca_certs'] = certifi.where()

            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,  # Automatically decode responses to strings
                socket_timeout=10,  # Increased timeout for remote Redis
                socket_connect_timeout=10,
                retry_on_timeout=True,  # Auto-retry on timeout
                health_check_interval=30,  # Ping every 30s to detect stale connections
                **ssl_kwargs,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {str(e)}")
            raise

    def _execute_with_retry(self, operation, max_retries: int = 3):
        """Execute a Redis operation with retry logic and exponential backoff"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return operation()
            except (redis.ConnectionError, redis.TimeoutError) as e:
                last_error = e
                logger.warning(f"Redis operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.5s, 1s, 2s...
                    sleep_time = 0.5 * (2 ** attempt)
                    time.sleep(sleep_time)
                    # Try to reconnect
                    try:
                        self.redis_client.ping()
                    except Exception:
                        pass  # Will retry the operation anyway
        raise last_error

    def validate_license(self, license_key: str) -> Dict[str, Any]:
        """
        Validate a license key and return license details

        Args:
            license_key: The license key to validate

        Returns:
            dict with:
                - valid (bool): Whether the license is valid
                - user_id (str): User ID if valid, None otherwise
                - license (dict): Full license data if valid
                - error (str): Error message if invalid
        """
        try:
            # Direct lookup with retry - ONE Redis command!
            license_data = self._execute_with_retry(
                lambda: self.redis_client.hgetall(f"license_key:{license_key}")
            )

            if not license_data:
                logger.warning(f"License key not found: {license_key[:20]}...")
                return {
                    'valid': False,
                    'user_id': None,
                    'license': None,
                    'error': 'License key not found'
                }

            # Parse boolean and JSON fields
            is_valid = license_data.get('isValid', 'false') == 'true'
            status = license_data.get('status', 'inactive')
            expires_at = license_data.get('expires_at')
            # Admin-imposed editor block on the key's owner, denormalized onto the hash
            # by the backend (services/redisLicenseService.ts). Defaults to not-blocked
            # so hashes written before this field existed keep working.
            editor_blocked = license_data.get('editor_blocked', 'false') == 'true'

            # Check if expired
            is_expired = False
            if expires_at:
                expiry_date = parse_expiry_date(expires_at)
                if expiry_date:
                    now = datetime.now(timezone.utc) if expiry_date.tzinfo else datetime.now()
                    is_expired = expiry_date < now
                else:
                    logger.warning(f"Could not parse expiry date format: {expires_at}")

            # Validate license status
            if not is_valid:
                return {
                    'valid': False,
                    'user_id': None,
                    'license': None,
                    'error': 'License is not valid'
                }

            if status != 'active':
                return {
                    'valid': False,
                    'user_id': None,
                    'license': None,
                    'error': f'License is {status}'
                }

            if is_expired:
                return {
                    'valid': False,
                    'user_id': None,
                    'license': None,
                    'error': 'License has expired'
                }

            # The editor is reached directly by AI streaming calls that never touch the
            # Express backend, so this block has to be enforced here too — otherwise a
            # blocked user keeps burning platform-funded model credits.
            if editor_blocked:
                logger.warning(
                    f"Blocked editor access for license: {license_key[:20]}..."
                )
                return {
                    'valid': False,
                    'user_id': None,
                    'license': None,
                    # 'code' marks this as a complete, user-facing sentence so callers
                    # present it as-is instead of wrapping it in "Invalid license: ".
                    'code': 'EDITOR_ACCESS_BLOCKED',
                    'error': 'Your access to the Eddyter editor has been suspended by '
                             'an administrator. Your account is still active — contact '
                             'support@eddyter.com to appeal.'
                }

            # Parse customizations
            customizations = {}
            if 'customizations' in license_data:
                try:
                    customizations = json.loads(license_data['customizations'])
                except Exception as e:
                    logger.warning(f"Failed to parse customizations: {e}")

            user_id = license_data.get('user_id')

            logger.info(f"✅ License validated successfully for user: {user_id}")

            return {
                'valid': True,
                'user_id': user_id,
                'license': {
                    'id': license_data.get('id'),
                    'user_id': user_id,
                    'licenseKey': license_key,
                    'isValid': is_valid,
                    'status': status,
                    'expires_at': expires_at,
                    'paid_plan': license_data.get('paid_plan', 'false') == 'true',
                    'customizations': customizations,
                    'is_live_api': license_data.get('is_live_api', 'false') == 'true',
                    'api_domain': license_data.get('api_domain'),
                    'subscription_status': license_data.get('subscription_status', ''),
                    'requires_user_api_keys': license_data.get('requires_user_api_keys', 'false') == 'true',
                },
                'error': None
            }

        except redis.RedisError as e:
            logger.error(f"Redis error during license validation: {str(e)}")
            return {
                'valid': False,
                'user_id': None,
                'license': None,
                'error': 'Database error occurred'
            }
        except Exception as e:
            logger.error(f"Unexpected error during license validation: {str(e)}")
            return {
                'valid': False,
                'user_id': None,
                'license': None,
                'error': 'Validation error occurred'
            }

    def get_user_api_keys(self, user_id: str) -> Dict[str, Optional[str]]:
        """
        Get user's API keys from Redis and decrypt them

        Args:
            user_id: The user ID to lookup

        Returns:
            dict with decrypted API keys or None values if not found:
                - use_own_keys (bool): Whether user uses their own keys
                - openai_api_key (str|None): Decrypted OpenAI key
                - anthropic_api_key (str|None): Decrypted Anthropic key
                - grok_api_key (str|None): Decrypted Grok key
                - gemini_api_key (str|None): Decrypted Gemini key
                - assemblyai_api_key (str|None): Decrypted AssemblyAI key
        """
        try:
            # Direct lookup with retry - ONE Redis command!
            api_keys_data = self._execute_with_retry(
                lambda: self.redis_client.hgetall(f"user_api_keys:{user_id}")
            )

            if not api_keys_data:
                logger.info(f"No API keys found for user: {user_id}")
                return {
                    'use_own_keys': False,
                    'openai_api_key': None,
                    'anthropic_api_key': None,
                    'grok_api_key': None,
                    'gemini_api_key': None,
                    'assemblyai_api_key': None,
                }

            use_own_keys = api_keys_data.get('use_own_keys', 'false') == 'true'

            # Decrypt API keys if they exist
            def decrypt_key(key_name: str) -> Optional[str]:
                encrypted = api_keys_data.get(key_name)
                if not encrypted or encrypted == 'null':
                    return None
                try:
                    decrypted = decrypt_api_key(encrypted)
                    if decrypted:
                        logger.debug(f"Successfully decrypted {key_name} for user {user_id}")
                    return decrypted or None
                except Exception as e:
                    logger.error(f"Failed to decrypt {key_name} for user {user_id}: {str(e)}")
                    return None

            result = {
                'use_own_keys': use_own_keys,
                'openai_api_key': decrypt_key('openai_api_key'),
                'anthropic_api_key': decrypt_key('anthropic_api_key'),
                'grok_api_key': decrypt_key('grok_api_key'),
                'gemini_api_key': decrypt_key('gemini_api_key'),
                'assemblyai_api_key': decrypt_key('assemblyai_api_key'),
            }

            logger.info(f"✅ API keys retrieved for user: {user_id}, use_own_keys: {use_own_keys}")

            return result

        except redis.RedisError as e:
            logger.error(f"Redis error while fetching API keys: {str(e)}")
            return {
                'use_own_keys': False,
                'openai_api_key': None,
                'anthropic_api_key': None,
                'grok_api_key': None,
                'gemini_api_key': None,
                'assemblyai_api_key': None,
            }
        except Exception as e:
            logger.error(f"Unexpected error while fetching API keys: {str(e)}")
            return {
                'use_own_keys': False,
                'openai_api_key': None,
                'anthropic_api_key': None,
                'grok_api_key': None,
                'gemini_api_key': None,
                'assemblyai_api_key': None,
            }

    def get_api_key_for_model(
        self,
        user_id: str,
        model_provider: str,
        default_keys: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Get the appropriate API key for a model provider
        Uses user's key if available, otherwise falls back to default

        Args:
            user_id: The user ID
            model_provider: Model provider (openai, claude, grok, gemini, deepseek)
            default_keys: Dictionary of default API keys from environment

        Returns:
            API key string or None if not found
        """
        # Get user's API keys
        user_keys = self.get_user_api_keys(user_id)

        # Map model providers to key names
        key_mapping = {
            'openai': 'openai_api_key',
            'claude': 'anthropic_api_key',
            'grok': 'grok_api_key',
            'gemini': 'gemini_api_key',
            'deepseek': 'openai_api_key',  # DeepSeek uses OpenAI-compatible API
        }

        key_field = key_mapping.get(model_provider.lower())

        if not key_field:
            logger.warning(f"Unknown model provider: {model_provider}")
            return None

        # Use user's key if they have one and use_own_keys is true
        if user_keys['use_own_keys'] and user_keys.get(key_field):
            logger.info(f"Using user's own API key for {model_provider}")
            return user_keys[key_field]

        # Fall back to default key
        if default_keys and key_field in default_keys:
            logger.info(f"Using default API key for {model_provider}")
            return default_keys[key_field]

        logger.warning(f"No API key found for {model_provider}")
        return None

    def update_heartbeat(self, license_key: str) -> bool:
        """
        Update the last_heartbeat timestamp for a license

        Args:
            license_key: The license key to update

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = self.redis_client.hset(
                f"license_key:{license_key}",
                'last_heartbeat',
                now
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {str(e)}")
            return False

    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")


# Singleton instance
_redis_manager = None


def get_redis_manager(force_reconnect: bool = False) -> RedisManager:
    """
    Get or create singleton Redis manager instance

    Args:
        force_reconnect: If True, create a new connection even if one exists

    Returns:
        RedisManager instance
    """
    global _redis_manager
    if _redis_manager is None or force_reconnect:
        if _redis_manager is not None:
            try:
                _redis_manager.close()
            except Exception:
                pass
        _redis_manager = RedisManager()
    return _redis_manager


def reset_redis_manager():
    """Reset the Redis manager singleton (useful for reconnection)"""
    global _redis_manager
    if _redis_manager is not None:
        try:
            _redis_manager.close()
        except Exception:
            pass
        _redis_manager = None

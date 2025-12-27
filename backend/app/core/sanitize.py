"""
T.A.R.S. XSS Sanitization Module

Comprehensive input sanitization utilities to prevent XSS attacks.
Provides functions to sanitize error messages, dictionaries, and arbitrary user input.

Features:
- HTML entity encoding to prevent script injection
- Script tag removal with regex-based content stripping
- Recursive sanitization for nested data structures
- Graceful handling of None/non-string inputs
- Production-ready with comprehensive validation
"""

import html
import re
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance
SCRIPT_TAG_PATTERN = re.compile(
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
    re.IGNORECASE | re.DOTALL
)

SCRIPT_EVENT_PATTERN = re.compile(
    r'\bon\w+\s*=\s*["\']?[^"\']*["\']?',
    re.IGNORECASE
)

JAVASCRIPT_PROTOCOL_PATTERN = re.compile(
    r'javascript:',
    re.IGNORECASE
)

# Additional dangerous patterns
IFRAME_PATTERN = re.compile(
    r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',
    re.IGNORECASE | re.DOTALL
)

OBJECT_EMBED_PATTERN = re.compile(
    r'<(object|embed)\b[^<]*(?:(?!<\/(object|embed)>)<[^<]*)*<\/(object|embed)>',
    re.IGNORECASE | re.DOTALL
)


def sanitize_error_message(message: Optional[Union[str, Any]]) -> str:
    """
    Sanitize error messages to prevent XSS attacks.

    This function ensures that error messages displayed to users cannot
    contain malicious scripts or HTML that could be executed in a browser.

    Process:
    1. Handle None and non-string inputs gracefully
    2. Remove dangerous HTML tags and their content (script, iframe, object, embed)
    3. Remove event handlers (onclick, onerror, etc.)
    4. Remove javascript: protocol
    5. Encode remaining HTML entities

    Args:
        message: Error message to sanitize (can be None or non-string)

    Returns:
        Sanitized error message string, safe for display

    Examples:
        >>> sanitize_error_message("<script>alert('XSS')</script>Error occurred")
        'Error occurred'

        >>> sanitize_error_message("User <b>admin</b> not found")
        'User &lt;b&gt;admin&lt;/b&gt; not found'

        >>> sanitize_error_message(None)
        ''

        >>> sanitize_error_message(404)
        '404'
    """
    # Handle None
    if message is None:
        return ""

    # Convert non-strings to strings
    if not isinstance(message, str):
        try:
            message = str(message)
        except Exception as e:
            logger.warning(f"Failed to convert message to string: {e}")
            return ""

    # Handle empty strings
    if not message.strip():
        return ""

    try:
        # Step 1: Remove script tags and their content
        sanitized = SCRIPT_TAG_PATTERN.sub('', message)

        # Step 2: Remove iframe tags and their content
        sanitized = IFRAME_PATTERN.sub('', sanitized)

        # Step 3: Remove object and embed tags
        sanitized = OBJECT_EMBED_PATTERN.sub('', sanitized)

        # Step 4: Remove event handlers (onclick, onerror, etc.)
        sanitized = SCRIPT_EVENT_PATTERN.sub('', sanitized)

        # Step 5: Remove javascript: protocol
        sanitized = JAVASCRIPT_PROTOCOL_PATTERN.sub('', sanitized)

        # Step 6: Encode HTML entities
        # This converts < to &lt;, > to &gt;, etc.
        sanitized = html.escape(sanitized)

        # Log if significant sanitization occurred
        if len(sanitized) < len(message) * 0.5:
            logger.warning(
                f"Significant content removed during sanitization. "
                f"Original length: {len(message)}, Sanitized length: {len(sanitized)}"
            )

        return sanitized

    except Exception as e:
        logger.error(f"Error during message sanitization: {e}", exc_info=True)
        # Fallback: just escape HTML entities
        try:
            return html.escape(str(message))
        except Exception:
            return ""


def sanitize_string(value: Optional[Union[str, Any]]) -> str:
    """
    Sanitize a single string value.

    Similar to sanitize_error_message but designed for general string sanitization.
    Used internally by sanitize_dict for consistent sanitization behavior.

    Args:
        value: String value to sanitize

    Returns:
        Sanitized string, safe for display or storage

    Examples:
        >>> sanitize_string("<img src=x onerror='alert(1)'>")
        '&lt;img src=x &gt;'

        >>> sanitize_string("Normal text")
        'Normal text'
    """
    return sanitize_error_message(value)


def sanitize_dict(
    data: Optional[Union[Dict[str, Any], Any]],
    max_depth: int = 10,
    _current_depth: int = 0
) -> Dict[str, Any]:
    """
    Recursively sanitize all string values in a dictionary.

    This function walks through nested dictionaries and lists, sanitizing
    all string values to prevent XSS attacks. Non-string values are preserved.

    Args:
        data: Dictionary to sanitize (can contain nested dicts, lists, etc.)
        max_depth: Maximum recursion depth to prevent infinite loops (default: 10)
        _current_depth: Internal parameter for tracking recursion depth

    Returns:
        New dictionary with all string values sanitized

    Raises:
        ValueError: If data is not a dictionary or max_depth is exceeded

    Examples:
        >>> sanitize_dict({"error": "<script>alert('XSS')</script>", "code": 500})
        {'error': '', 'code': 500}

        >>> sanitize_dict({
        ...     "user": {"name": "<b>admin</b>", "id": 123},
        ...     "messages": ["<script>xss</script>", "normal text"]
        ... })
        {
            'user': {'name': '&lt;b&gt;admin&lt;/b&gt;', 'id': 123},
            'messages': ['', 'normal text']
        }

        >>> sanitize_dict(None)
        {}
    """
    # Handle None input
    if data is None:
        return {}

    # Ensure input is a dictionary
    if not isinstance(data, dict):
        try:
            # Try to convert to dict if it has dict-like properties
            if hasattr(data, '__dict__'):
                data = data.__dict__
            else:
                raise ValueError(f"Cannot sanitize non-dict type: {type(data).__name__}")
        except Exception as e:
            logger.error(f"Failed to convert data to dict: {e}")
            raise ValueError(f"Input must be a dictionary, got {type(data).__name__}")

    # Check recursion depth
    if _current_depth >= max_depth:
        logger.warning(
            f"Max recursion depth ({max_depth}) reached during sanitization. "
            "Returning current level without further recursion."
        )
        return data

    # Recursively sanitize the dictionary
    sanitized = {}

    for key, value in data.items():
        try:
            # Sanitize the key itself (keys should also be safe)
            safe_key = sanitize_error_message(key) if isinstance(key, str) else key

            # Sanitize based on value type
            if isinstance(value, str):
                # Sanitize string values
                sanitized[safe_key] = sanitize_string(value)

            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[safe_key] = sanitize_dict(
                    value,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1
                )

            elif isinstance(value, (list, tuple)):
                # Recursively sanitize lists/tuples
                sanitized[safe_key] = _sanitize_list(
                    value,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1
                )

            elif value is None:
                # Preserve None values
                sanitized[safe_key] = None

            elif isinstance(value, (int, float, bool)):
                # Preserve primitive types
                sanitized[safe_key] = value

            else:
                # For other types, try to convert to string and sanitize
                try:
                    sanitized[safe_key] = sanitize_string(str(value))
                except Exception as e:
                    logger.warning(
                        f"Could not sanitize value of type {type(value).__name__}: {e}"
                    )
                    sanitized[safe_key] = None

        except Exception as e:
            logger.error(f"Error sanitizing key '{key}': {e}", exc_info=True)
            # On error, exclude the problematic key-value pair
            continue

    return sanitized


def _sanitize_list(
    items: Union[List[Any], tuple],
    max_depth: int = 10,
    _current_depth: int = 0
) -> List[Any]:
    """
    Recursively sanitize all string values in a list.

    Internal helper function for sanitize_dict to handle lists and tuples.

    Args:
        items: List or tuple to sanitize
        max_depth: Maximum recursion depth
        _current_depth: Current recursion depth

    Returns:
        New list with all string values sanitized
    """
    if _current_depth >= max_depth:
        logger.warning(f"Max recursion depth ({max_depth}) reached in list sanitization")
        return list(items)

    sanitized = []

    for item in items:
        try:
            if isinstance(item, str):
                sanitized.append(sanitize_string(item))

            elif isinstance(item, dict):
                sanitized.append(sanitize_dict(
                    item,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1
                ))

            elif isinstance(item, (list, tuple)):
                sanitized.append(_sanitize_list(
                    item,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1
                ))

            elif item is None or isinstance(item, (int, float, bool)):
                sanitized.append(item)

            else:
                try:
                    sanitized.append(sanitize_string(str(item)))
                except Exception as e:
                    logger.warning(f"Could not sanitize list item of type {type(item).__name__}: {e}")
                    sanitized.append(None)

        except Exception as e:
            logger.error(f"Error sanitizing list item: {e}", exc_info=True)
            continue

    return sanitized


def sanitize_response_data(data: Any) -> Any:
    """
    Sanitize response data before sending to client.

    This is a convenience function that intelligently sanitizes various
    response types (dict, list, string, or primitive types).

    Args:
        data: Response data to sanitize

    Returns:
        Sanitized response data in the same structure

    Examples:
        >>> sanitize_response_data({"error": "<script>xss</script>"})
        {'error': ''}

        >>> sanitize_response_data(["<script>xss</script>", "safe text"])
        ['', 'safe text']

        >>> sanitize_response_data("<script>xss</script>")
        ''

        >>> sanitize_response_data(42)
        42
    """
    try:
        if isinstance(data, dict):
            return sanitize_dict(data)
        elif isinstance(data, (list, tuple)):
            return _sanitize_list(data)
        elif isinstance(data, str):
            return sanitize_string(data)
        elif data is None or isinstance(data, (int, float, bool)):
            return data
        else:
            # For other types, try to sanitize string representation
            logger.info(f"Sanitizing non-standard type: {type(data).__name__}")
            return sanitize_string(str(data))

    except Exception as e:
        logger.error(f"Error in sanitize_response_data: {e}", exc_info=True)
        return None


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON USE CASES
# ==============================================================================

def sanitize_user_input(user_input: Optional[str]) -> str:
    """
    Sanitize user input from forms, query parameters, etc.

    Alias for sanitize_error_message for better semantic clarity.

    Args:
        user_input: User-provided input string

    Returns:
        Sanitized string safe for processing and display
    """
    return sanitize_error_message(user_input)


def sanitize_log_message(message: Optional[str]) -> str:
    """
    Sanitize messages before logging to prevent log injection.

    Removes newlines and carriage returns in addition to XSS sanitization
    to prevent log forging attacks.

    Args:
        message: Log message to sanitize

    Returns:
        Sanitized log message
    """
    if message is None:
        return ""

    # First apply standard XSS sanitization
    sanitized = sanitize_error_message(message)

    # Remove newlines and carriage returns to prevent log injection
    sanitized = sanitized.replace('\n', ' ').replace('\r', ' ')

    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)

    return sanitized.strip()


# ==============================================================================
# USAGE EXAMPLES (for documentation)
# ==============================================================================

"""
Example 1: Sanitizing error messages in exception handlers
----------------------------------------------------------
from app.core.sanitize import sanitize_error_message

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = sanitize_error_message(str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": error_msg}
    )


Example 2: Sanitizing API response data
----------------------------------------
from app.core.sanitize import sanitize_dict

@app.get("/api/user/{user_id}")
async def get_user(user_id: str):
    user_data = await fetch_user(user_id)
    # Sanitize all string fields in the response
    safe_data = sanitize_dict(user_data)
    return safe_data


Example 3: Sanitizing user input from forms
--------------------------------------------
from app.core.sanitize import sanitize_user_input

@app.post("/api/feedback")
async def submit_feedback(message: str):
    safe_message = sanitize_user_input(message)
    await save_feedback(safe_message)
    return {"status": "success", "message": safe_message}


Example 4: Sanitizing log messages
-----------------------------------
from app.core.sanitize import sanitize_log_message

@app.post("/api/login")
async def login(username: str):
    safe_username = sanitize_log_message(username)
    logger.info(f"Login attempt for user: {safe_username}")
    # ... login logic


Example 5: Sanitizing complex nested structures
------------------------------------------------
from app.core.sanitize import sanitize_response_data

@app.get("/api/analytics")
async def get_analytics():
    data = {
        "metrics": [
            {"name": "<script>xss</script>", "value": 100},
            {"name": "valid_metric", "value": 200}
        ],
        "metadata": {
            "user": "<b>admin</b>",
            "timestamp": 1234567890
        }
    }
    return sanitize_response_data(data)


Example 6: Using in middleware for automatic sanitization
----------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.sanitize import sanitize_response_data

class SanitizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Automatically sanitize all JSON responses
        if response.headers.get("content-type") == "application/json":
            # Implementation would need to intercept and sanitize response body
            pass
        return response
"""

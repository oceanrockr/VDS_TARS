"""
T.A.R.S. XSS Sanitization Integration Example

This file demonstrates how to integrate the sanitize module into
existing T.A.R.S. endpoints for XSS protection.

IMPORTANT: This is an example file for reference only.
Copy these patterns into your actual endpoint files.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
import logging

# Import sanitization functions
from app.core import (
    sanitize_error_message,
    sanitize_dict,
    sanitize_user_input,
    sanitize_log_message,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# EXAMPLE 1: Global Exception Handler with Sanitization
# ==============================================================================

app = FastAPI(title="T.A.R.S. API")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that sanitizes all error messages
    before returning them to the client.
    """
    # Sanitize the error message to prevent XSS
    safe_error = sanitize_error_message(str(exc))

    # Log the error (sanitized to prevent log injection)
    safe_path = sanitize_log_message(str(request.url.path))
    logger.error(f"Error on path {safe_path}: {sanitize_log_message(str(exc))}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": safe_error,
            "path": str(request.url.path)
        }
    )


# ==============================================================================
# EXAMPLE 2: Pydantic Model with Automatic Sanitization
# ==============================================================================

class UserProfileUpdate(BaseModel):
    """User profile update model with automatic sanitization"""
    bio: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None

    @validator('bio', 'website', 'location')
    def sanitize_string_fields(cls, v):
        """Automatically sanitize all string fields"""
        if v is None:
            return v
        return sanitize_user_input(v)


@app.put("/api/user/profile")
async def update_profile(profile: UserProfileUpdate, user_id: int):
    """
    Update user profile with automatic sanitization via Pydantic.
    All string fields are sanitized before reaching the business logic.
    """
    try:
        # Profile data is already sanitized by Pydantic validators
        updated = await update_user_profile(user_id, profile.dict())

        # Sanitize response data before returning
        return sanitize_dict(updated)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# EXAMPLE 3: Comment/Post Endpoint with XSS Protection
# ==============================================================================

class CommentCreate(BaseModel):
    """Comment creation model"""
    post_id: int
    content: str
    author: str

    @validator('content', 'author')
    def sanitize_user_content(cls, v):
        """Sanitize user-generated content"""
        return sanitize_user_input(v)


@app.post("/api/comments")
async def create_comment(comment: CommentCreate):
    """
    Create a comment with XSS protection.
    Input is sanitized via Pydantic, output is sanitized before returning.
    """
    try:
        # Save comment (already sanitized by Pydantic)
        saved_comment = await save_comment(comment.dict())

        # Sanitize response
        safe_response = sanitize_dict({
            "id": saved_comment.id,
            "content": saved_comment.content,
            "author": saved_comment.author,
            "created_at": saved_comment.created_at.isoformat()
        })

        return safe_response

    except Exception as e:
        # Log with sanitization
        logger.error(
            f"Comment creation failed: {sanitize_log_message(str(e))}"
        )
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


@app.get("/api/comments/{post_id}")
async def get_comments(post_id: int):
    """
    Fetch comments with XSS protection.
    All comments are sanitized before being returned to the client.
    """
    try:
        # Fetch comments from database
        comments = await fetch_comments_by_post(post_id)

        # Convert to dict and sanitize
        comments_data = [comment.to_dict() for comment in comments]

        # Sanitize the entire response
        safe_response = sanitize_dict({
            "post_id": post_id,
            "comments": comments_data,
            "total": len(comments_data)
        })

        return safe_response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# EXAMPLE 4: Search Endpoint with Input/Output Sanitization
# ==============================================================================

@app.get("/api/search")
async def search(query: str, limit: int = 10):
    """
    Search endpoint with sanitization of both input and output.
    """
    # Sanitize search query
    safe_query = sanitize_user_input(query)

    # Log the search (sanitized)
    logger.info(f"Search query: {sanitize_log_message(safe_query)}")

    try:
        # Perform search with sanitized query
        results = await perform_search(safe_query, limit)

        # Sanitize search results before returning
        safe_results = sanitize_dict({
            "query": safe_query,
            "results": results,
            "count": len(results)
        })

        return safe_results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# EXAMPLE 5: User Feedback Endpoint
# ==============================================================================

class FeedbackCreate(BaseModel):
    """Feedback submission model"""
    subject: str
    message: str
    email: Optional[str] = None

    @validator('subject', 'message', 'email')
    def sanitize_feedback(cls, v):
        """Sanitize all text fields"""
        if v is None:
            return v
        return sanitize_user_input(v)


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackCreate):
    """
    Submit user feedback with XSS protection.
    """
    try:
        # Save feedback (already sanitized)
        result = await save_feedback(feedback.dict())

        # Log feedback submission (sanitized)
        logger.info(
            f"Feedback submitted - Subject: {sanitize_log_message(feedback.subject)}"
        )

        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "id": result.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# EXAMPLE 6: Analytics/Metrics Endpoint
# ==============================================================================

@app.get("/api/analytics/user/{user_id}")
async def get_user_analytics(user_id: int):
    """
    Get user analytics with sanitized user-generated metadata.
    """
    try:
        # Fetch analytics data
        analytics = await fetch_user_analytics(user_id)

        # Sanitize the entire response (especially user metadata)
        safe_analytics = sanitize_dict({
            "user_id": user_id,
            "metrics": analytics.metrics,
            "metadata": analytics.metadata,  # May contain user-generated data
            "generated_at": analytics.timestamp.isoformat()
        })

        return safe_analytics

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# EXAMPLE 7: Batch Processing with Sanitization
# ==============================================================================

@app.post("/api/batch/import")
async def batch_import(items: list[Dict[str, Any]]):
    """
    Batch import endpoint that sanitizes all items.
    """
    try:
        # Sanitize all items before processing
        safe_items = [sanitize_dict(item) for item in items]

        # Process items
        results = await process_batch_import(safe_items)

        # Sanitize results
        return sanitize_dict({
            "processed": len(safe_items),
            "succeeded": results.success_count,
            "failed": results.failure_count,
            "errors": results.errors
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(str(e))
        )


# ==============================================================================
# HELPER FUNCTIONS (Mock implementations)
# ==============================================================================

async def update_user_profile(user_id: int, profile_data: dict):
    """Mock function - implement in your actual service"""
    pass


async def save_comment(comment_data: dict):
    """Mock function - implement in your actual service"""
    pass


async def fetch_comments_by_post(post_id: int):
    """Mock function - implement in your actual service"""
    pass


async def perform_search(query: str, limit: int):
    """Mock function - implement in your actual service"""
    pass


async def save_feedback(feedback_data: dict):
    """Mock function - implement in your actual service"""
    pass


async def fetch_user_analytics(user_id: int):
    """Mock function - implement in your actual service"""
    pass


async def process_batch_import(items: list):
    """Mock function - implement in your actual service"""
    pass


# ==============================================================================
# BEST PRACTICES SUMMARY
# ==============================================================================

"""
1. INPUT SANITIZATION
   - Use Pydantic validators to sanitize user input automatically
   - Sanitize query parameters and path parameters
   - Always sanitize before business logic

2. OUTPUT SANITIZATION
   - Sanitize all response data using sanitize_dict()
   - Pay special attention to user-generated content
   - Sanitize error messages before returning to client

3. LOGGING
   - Always use sanitize_log_message() for user input in logs
   - Prevent log injection attacks
   - Never log raw user input

4. ERROR HANDLING
   - Use global exception handler with sanitization
   - Sanitize all exception messages
   - Log sanitized errors for debugging

5. TESTING
   - Test with XSS attack vectors
   - Verify sanitization doesn't break functionality
   - Check edge cases (None, empty strings, etc.)

6. PERFORMANCE
   - Sanitization is fast but not free
   - Consider caching sanitized values for frequently accessed data
   - Only sanitize user-generated content, not system data

7. DEFENSE IN DEPTH
   - Sanitization is one layer
   - Also use CSP headers, HTTP-only cookies, CORS
   - Combine with other security middleware
"""

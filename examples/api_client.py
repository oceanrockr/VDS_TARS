#!/usr/bin/env python3
"""
examples/api_client.py
Official example Python client for interacting with the T.A.R.S. Enterprise API.

Supports:
- API key authentication
- JWT authentication (login + token refresh)
- All major endpoints (health, metrics, GA KPI, daily summaries, anomalies, regressions, retrospective)
- Error handling with exponential backoff retry
- Response validation
- Pretty-printed output

Compatible with Phase 14.6 (v1.0.2-RC1)

Usage Examples:

    # Health check (no auth)
    python examples/api_client.py --mode health

    # Get GA KPIs with API key
    python examples/api_client.py --api-key tars_admin_default_key_change_in_prod --mode ga

    # Get all data with JWT
    python examples/api_client.py --username admin --password admin123 --mode all

    # Custom API URL
    python examples/api_client.py --url https://tars.example.com --username admin --password admin123 --mode daily
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


# ============================================================================
# T.A.R.S. API CLIENT
# ============================================================================

class TARSClient:
    """
    Python client for the T.A.R.S. Enterprise API.

    Handles authentication (API key or JWT), request retry logic,
    and response validation for all endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Initialize the T.A.R.S. API client.

        Args:
            base_url: Base URL of the API server (default: http://localhost:8100)
            api_key: API key for authentication (X-API-Key header)
            username: Username for JWT authentication
            password: Password for JWT authentication
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.timeout = timeout
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

        # Validate authentication
        if not api_key and not (username and password):
            print("WARNING: No authentication credentials provided. Only public endpoints will work.")

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        require_auth: bool = True,
    ) -> requests.Response:
        """
        Make an HTTP request with retry logic and authentication.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., '/api/ga')
            params: Query parameters
            data: Form data
            json_data: JSON body
            retries: Number of retry attempts for 5xx and 429 errors
            require_auth: Whether endpoint requires authentication

        Returns:
            requests.Response object

        Raises:
            requests.HTTPError: On 4xx errors (after all retries exhausted for 5xx/429)
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}

        # Add authentication
        if require_auth:
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            elif self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            elif self.username and self.password:
                # Need to authenticate first
                self.authenticate_jwt()
                headers['Authorization'] = f'Bearer {self.access_token}'

        # Retry loop
        for attempt in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Success
                if response.status_code < 400:
                    return response

                # Rate limited - wait and retry
                if response.status_code == 429:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Rate limited (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    continue

                # Server error - retry
                if response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"Server error ({response.status_code}). Retrying in {wait_time}s... (attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    continue

                # Client error - don't retry
                response.raise_for_status()

            except requests.Timeout:
                if attempt < retries - 1:
                    print(f"Request timeout. Retrying... (attempt {attempt + 1}/{retries})")
                    continue
                raise
            except requests.ConnectionError:
                if attempt < retries - 1:
                    print(f"Connection error. Retrying... (attempt {attempt + 1}/{retries})")
                    time.sleep(2 ** attempt)
                    continue
                raise

        # All retries exhausted
        response.raise_for_status()
        return response

    # ========================================================================
    # AUTHENTICATION
    # ========================================================================

    def authenticate_jwt(self) -> str:
        """
        Authenticate with username/password and get JWT tokens.

        Returns:
            Access token

        Raises:
            ValueError: If username/password not provided
            requests.HTTPError: If authentication fails
        """
        if not self.username or not self.password:
            raise ValueError("Username and password required for JWT authentication")

        print(f"Authenticating as {self.username}...")

        response = self._request(
            'POST',
            '/auth/login',
            json_data={
                'username': self.username,
                'password': self.password,
            },
            require_auth=False,
        )

        auth_data = response.json()

        if 'access_token' not in auth_data:
            raise ValueError("Missing access_token in authentication response")

        self.access_token = auth_data['access_token']
        self.refresh_token = auth_data.get('refresh_token')

        print(f"✓ Authenticated successfully (token expires in {auth_data.get('expires_in', 'unknown')} seconds)")

        return self.access_token

    def refresh_access_token(self) -> str:
        """
        Refresh the access token using the refresh token.

        Returns:
            New access token

        Raises:
            ValueError: If refresh token not available
            requests.HTTPError: If refresh fails
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available. Authenticate first.")

        print("Refreshing access token...")

        response = self._request(
            'POST',
            '/auth/refresh',
            json_data={'refresh_token': self.refresh_token},
            require_auth=False,
        )

        auth_data = response.json()
        self.access_token = auth_data['access_token']

        print("✓ Access token refreshed")

        return self.access_token

    # ========================================================================
    # PUBLIC ENDPOINTS (no authentication required)
    # ========================================================================

    def get_health(self) -> Dict[str, Any]:
        """
        Get API health status.

        Returns:
            Health status dictionary
        """
        response = self._request('GET', '/health', require_auth=False)
        health = response.json()

        # Validate response
        if 'status' not in health:
            raise ValueError("Invalid health response: missing 'status' field")

        return health

    def get_metrics(self) -> str:
        """
        Get Prometheus metrics.

        Returns:
            Prometheus metrics in text format
        """
        response = self._request('GET', '/metrics', require_auth=False)
        return response.text

    # ========================================================================
    # PROTECTED ENDPOINTS (authentication required)
    # ========================================================================

    def get_ga_kpi(self) -> Dict[str, Any]:
        """
        Get General Availability KPIs.

        Returns:
            GA KPI dictionary with metrics

        Raises:
            ValueError: If response is missing required fields
        """
        response = self._request('GET', '/api/ga')
        ga_data = response.json()

        # Validate response
        required_fields = ['timestamp', 'kpis']
        missing = [f for f in required_fields if f not in ga_data]
        if missing:
            raise ValueError(f"Invalid GA KPI response: missing fields {missing}")

        return ga_data

    def list_daily_summaries(self, limit: int = 7) -> List[Dict[str, Any]]:
        """
        Get list of daily summaries (last N days).

        Args:
            limit: Number of days to retrieve (default: 7)

        Returns:
            List of daily summary dictionaries
        """
        response = self._request('GET', '/api/daily', params={'limit': limit})
        summaries = response.json()

        if not isinstance(summaries, list):
            raise ValueError("Invalid daily summaries response: expected list")

        return summaries

    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """
        Get daily summary for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Daily summary dictionary

        Raises:
            ValueError: If response is missing required fields
        """
        response = self._request('GET', f'/api/daily/{date}')
        summary = response.json()

        # Validate response
        required_fields = ['date', 'availability', 'error_count']
        missing = [f for f in required_fields if f not in summary]
        if missing:
            raise ValueError(f"Invalid daily summary response: missing fields {missing}")

        return summary

    def get_anomalies(
        self,
        severity: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get detected anomalies.

        Args:
            severity: Filter by severity (critical, warning, info)
            date: Get anomalies for specific date (YYYY-MM-DD)
            limit: Maximum number of anomalies to return

        Returns:
            List of anomaly dictionaries
        """
        params = {'limit': limit}
        if severity:
            params['severity'] = severity

        endpoint = f'/api/anomalies/{date}' if date else '/api/anomalies'
        response = self._request('GET', endpoint, params=params)
        anomalies = response.json()

        if not isinstance(anomalies, list):
            raise ValueError("Invalid anomalies response: expected list")

        return anomalies

    def get_regressions(self, date: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get detected performance regressions.

        Args:
            date: Get regressions for specific date (YYYY-MM-DD)
            limit: Maximum number of regressions to return

        Returns:
            List of regression dictionaries
        """
        params = {'limit': limit}
        endpoint = f'/api/regressions/{date}' if date else '/api/regressions'
        response = self._request('GET', endpoint, params=params)
        regressions = response.json()

        if not isinstance(regressions, list):
            raise ValueError("Invalid regressions response: expected list")

        return regressions

    def generate_retrospective(self, output_format: str = 'json') -> Dict[str, Any]:
        """
        Generate a retrospective report.

        Args:
            output_format: Output format (json or markdown)

        Returns:
            Retrospective data or job information

        Raises:
            ValueError: If format is invalid
        """
        if output_format not in ['json', 'markdown']:
            raise ValueError(f"Invalid format: {output_format}. Must be 'json' or 'markdown'")

        response = self._request(
            'POST',
            '/api/retrospective',
            json_data={'format': output_format}
        )

        return response.json()

    def download_retrospective(self, filename: str, output_path: Optional[Path] = None) -> Path:
        """
        Download a generated retrospective file.

        Args:
            filename: Filename to download
            output_path: Local path to save file (default: current directory)

        Returns:
            Path to downloaded file
        """
        response = self._request('GET', f'/api/retrospective/download/{filename}')

        if output_path is None:
            output_path = Path(filename)

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return output_path


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_json(data: Any, indent: int = 2) -> None:
    """Pretty-print JSON data."""
    print(json.dumps(data, indent=indent, default=str))


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    parser = argparse.ArgumentParser(
        description="T.A.R.S. Enterprise API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Health check (no auth)
  python api_client.py --mode health

  # Get GA KPIs with API key
  python api_client.py --api-key tars_admin_default_key_change_in_prod --mode ga

  # Get all data with JWT
  python api_client.py --username admin --password admin123 --mode all

  # Get daily summary for specific date
  python api_client.py --username admin --password admin123 --mode daily --date 2025-11-27
        """
    )

    parser.add_argument(
        '--url',
        default='http://localhost:8100',
        help='API base URL (default: http://localhost:8100)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for authentication (X-API-Key header)'
    )
    parser.add_argument(
        '--username',
        help='Username for JWT authentication'
    )
    parser.add_argument(
        '--password',
        help='Password for JWT authentication'
    )
    parser.add_argument(
        '--mode',
        choices=['health', 'metrics', 'ga', 'daily', 'anomalies', 'regressions', 'retro', 'all'],
        default='health',
        help='API endpoint to call (default: health)'
    )
    parser.add_argument(
        '--date',
        help='Date filter for daily/anomalies/regressions (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--severity',
        choices=['critical', 'warning', 'info'],
        help='Severity filter for anomalies'
    )

    args = parser.parse_args()

    try:
        # Create client
        client = TARSClient(
            base_url=args.url,
            api_key=args.api_key,
            username=args.username,
            password=args.password,
        )

        # Execute mode
        if args.mode == 'health':
            print_header("API Health Check")
            health = client.get_health()
            print_json(health)
            print(f"\n✓ Status: {health['status']}")

        elif args.mode == 'metrics':
            print_header("Prometheus Metrics")
            metrics = client.get_metrics()
            print(metrics)

        elif args.mode == 'ga':
            print_header("GA KPIs")
            ga_data = client.get_ga_kpi()
            print_json(ga_data)
            if 'kpis' in ga_data:
                print(f"\n✓ Retrieved {len(ga_data['kpis'])} KPIs")

        elif args.mode == 'daily':
            if args.date:
                print_header(f"Daily Summary for {args.date}")
                summary = client.get_daily_summary(args.date)
                print_json(summary)
            else:
                print_header("Daily Summaries (Last 7 Days)")
                summaries = client.list_daily_summaries()
                print_json(summaries)
                print(f"\n✓ Retrieved {len(summaries)} daily summaries")

        elif args.mode == 'anomalies':
            print_header("Anomalies")
            anomalies = client.get_anomalies(severity=args.severity, date=args.date)
            print_json(anomalies)
            print(f"\n✓ Found {len(anomalies)} anomalies")

        elif args.mode == 'regressions':
            print_header("Performance Regressions")
            regressions = client.get_regressions(date=args.date)
            print_json(regressions)
            print(f"\n✓ Found {len(regressions)} regressions")

        elif args.mode == 'retro':
            print_header("Generate Retrospective")
            result = client.generate_retrospective(output_format='json')
            print_json(result)
            print("\n✓ Retrospective generated")

        elif args.mode == 'all':
            # Run all endpoints
            print_header("Running All API Endpoints")

            print("\n1. Health Check")
            health = client.get_health()
            print(f"   Status: {health['status']}")

            print("\n2. GA KPIs")
            ga_data = client.get_ga_kpi()
            print(f"   KPIs: {len(ga_data.get('kpis', []))}")

            print("\n3. Daily Summaries")
            summaries = client.list_daily_summaries()
            print(f"   Summaries: {len(summaries)}")

            print("\n4. Anomalies")
            anomalies = client.get_anomalies()
            print(f"   Anomalies: {len(anomalies)}")

            print("\n5. Regressions")
            regressions = client.get_regressions()
            print(f"   Regressions: {len(regressions)}")

            print("\n✓ All endpoints tested successfully!")

        return 0

    except requests.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"   Status Code: {e.response.status_code}", file=sys.stderr)
            try:
                error_data = e.response.json()
                print(f"   Error: {json.dumps(error_data, indent=2)}", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"   Response: {e.response.text[:500]}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

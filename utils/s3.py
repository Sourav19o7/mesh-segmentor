"""
S3 client utilities for model and data storage.
"""

import os
import io
from pathlib import Path
from typing import Optional, BinaryIO, Union, List
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from utils.logging import get_logger

logger = get_logger(__name__)


class S3Client:
    """S3 client wrapper with retry logic and error handling."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        max_retries: int = 3,
    ):
        """
        Initialize S3 client.

        Args:
            bucket: Default S3 bucket name
            region: AWS region
            max_retries: Maximum number of retries for failed operations
        """
        self.bucket = bucket
        self.region = region

        boto_config = BotoConfig(
            region_name=region,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
        )

        self.client = boto3.client("s3", config=boto_config)
        self.resource = boto3.resource("s3", config=boto_config)

    def upload_file(
        self,
        local_path: Union[str, Path],
        s3_key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket: Optional bucket override
            content_type: Optional content type

        Returns:
            S3 URI (s3://bucket/key)
        """
        bucket = bucket or self.bucket
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")

        try:
            self.client.upload_file(
                str(local_path),
                bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )
            return f"s3://{bucket}/{s3_key}"
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload bytes directly to S3.

        Args:
            data: Bytes to upload
            s3_key: S3 object key
            bucket: Optional bucket override
            content_type: Optional content type

        Returns:
            S3 URI
        """
        bucket = bucket or self.bucket
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        logger.info(f"Uploading bytes to s3://{bucket}/{s3_key}")

        try:
            self.client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=data,
                **extra_args,
            )
            return f"s3://{bucket}/{s3_key}"
        except ClientError as e:
            logger.error(f"Failed to upload bytes: {e}")
            raise

    def download_file(
        self,
        s3_key: str,
        local_path: Union[str, Path],
        bucket: Optional[str] = None,
    ) -> Path:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Local destination path
            bucket: Optional bucket override

        Returns:
            Local file path
        """
        bucket = bucket or self.bucket
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")

        try:
            self.client.download_file(bucket, s3_key, str(local_path))
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def download_bytes(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """
        Download file content as bytes.

        Args:
            s3_key: S3 object key
            bucket: Optional bucket override

        Returns:
            File content as bytes
        """
        bucket = bucket or self.bucket

        logger.info(f"Downloading bytes from s3://{bucket}/{s3_key}")

        try:
            response = self.client.get_object(Bucket=bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def generate_presigned_url(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for S3 object access.

        Args:
            s3_key: S3 object key
            bucket: Optional bucket override
            expiration: URL expiration in seconds
            http_method: HTTP method (GET or PUT)

        Returns:
            Presigned URL
        """
        bucket = bucket or self.bucket

        try:
            client_method = (
                "get_object" if http_method == "GET" else "put_object"
            )
            url = self.client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def list_objects(
        self,
        prefix: str,
        bucket: Optional[str] = None,
    ) -> List[str]:
        """
        List objects with a given prefix.

        Args:
            prefix: S3 key prefix
            bucket: Optional bucket override

        Returns:
            List of S3 keys
        """
        bucket = bucket or self.bucket
        keys = []

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
            return keys
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}")
            raise

    def exists(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Check if an S3 object exists.

        Args:
            s3_key: S3 object key
            bucket: Optional bucket override

        Returns:
            True if object exists
        """
        bucket = bucket or self.bucket

        try:
            self.client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def delete(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
    ) -> None:
        """
        Delete an S3 object.

        Args:
            s3_key: S3 object key
            bucket: Optional bucket override
        """
        bucket = bucket or self.bucket

        try:
            self.client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Deleted s3://{bucket}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            raise


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.

    Args:
        uri: S3 URI (s3://bucket/key)

    Returns:
        Tuple of (bucket, key)
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

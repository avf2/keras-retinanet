import os
import re

import boto3
from collections import deque
from datetime import datetime


def digest_bucket_name(bucket_name):
    bucket_name_parts = bucket_name.split('/')
    bucket_name_parts = deque(bucket_name_parts)
    real_bucket_name = bucket_name_parts.popleft()
    base_key = ''
    while True:
        try:
            key = bucket_name_parts.pop()
            base_key += '/' + key
        except IndexError:
            break
    return real_bucket_name, base_key


def get_file_url(key_to_file, bucket_name):
    return "s3://{}/{}".format(bucket_name, key_to_file)


def read_file_content(key_to_file, bucket_name):
    file_object = boto3.resource('s3').Object(bucket_name, key_to_file)
    return file_object.get()['Body'].read().decode('utf-8')


def download_file_from_s3_url(frame_url, output_path=None, verbose=True):
    """
    :param frame_url: of the form  "s3://bucket_name/..."
    """
    url_pattern = 's3://(.*?)/(.*)'
    url_search = re.search(url_pattern, frame_url)
    try:
        bucket_name, key_to_file = url_search.groups()
    except AttributeError:
        raise ValueError('frame_url should be something like "s3://bucket_name/...", not {}'.format(frame_url))
    return download_file_from_bucket(key_to_file, bucket_name=bucket_name, output_path=output_path, verbose=verbose)


def download_file_from_bucket(file_path_inside_s3, bucket_name, output_path=None, verbose=True):
    if verbose:
        print('Downloading {} from bucket {} into file...'.format(file_path_inside_s3, bucket_name))
    if not output_path:
        output_path = os.path.join('/tmp', file_path_inside_s3)

    output_dir = os.path.split(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).download_file(file_path_inside_s3, output_path)
    if verbose:
        print('File {} downloaded from bucket {} and saved in {}.'
                    .format(file_path_inside_s3, bucket_name, output_path))
    return output_path


def upload_file_to_bucket(path_to_file, dir_in_s3, bucket_name, new_file_name=None, verbose=True):
    if verbose:
        print('Uploading file {} to {} in bucket {}...'.format(path_to_file, dir_in_s3, bucket_name))
    s3 = boto3.resource('s3')
    if new_file_name:
        path_in_s3 = os.path.join(dir_in_s3, new_file_name)
    else:
        file_name = os.path.basename(path_to_file)
        path_in_s3 = os.path.join(dir_in_s3, file_name)
    s3.Bucket(bucket_name).upload_file(path_to_file, path_in_s3)
    if verbose:
        print('File {} uploaded successfully to s3://{}/{}.'.format(path_to_file, bucket_name, path_in_s3))


def list_files_from_sample(sample_name, dir_in_bucket, bucket_name):
    sample_dir_path = os.path.join(dir_in_bucket, sample_name)
    sample_files = list_files_from_bucket_path(sample_dir_path, bucket_name=bucket_name, filename_prefix=None,
                                               filename_sufix=None, include_datetimes=False)
    sample_files = [os.path.basename(sf) for sf in sample_files if not sf.endswith('/')]
    return sample_files


def list_newer_samples_from_bucket_path(path_in_bucket, from_datetime, bucket_name):
    print('Listing recent samples (from {} on) from path {} (bucket {})...'
                 .format(str(from_datetime), path_in_bucket, bucket_name))
    newer_files = list_newer_files_from_bucket_path(path_in_bucket, from_datetime, bucket_name=bucket_name,
                                                    filename_prefix='VVPM', filename_sufix=None, verbose=False)
    newer_samples = [os.path.basename(os.path.split(nwf)[0]) for nwf in newer_files]
    sample_names = list(set(newer_samples))
    print('{} samples (from {} on) found in path {} from bucket {}.'
                .format(len(sample_names), str(from_datetime), path_in_bucket, bucket_name))
    return sample_names


def list_newer_files_from_bucket_path(path_in_bucket, from_datetime, bucket_name, filename_prefix=None,
                                      filename_sufix=None, verbose=True):
    if verbose:
        print('Listing recent files (from {} on) from path {} (bucket {})...'
                     .format(str(from_datetime), path_in_bucket, bucket_name))
    if not isinstance(from_datetime, datetime):
        from_datetime = datetime.strptime(from_datetime, '%Y/%m/%d-%H:%M:%S-UTC%z')
    files_and_dates = list_files_from_bucket_path(path_in_bucket, bucket_name, filename_prefix, filename_sufix, True)
    files = [f for (f, d) in files_and_dates if d >= from_datetime]
    if verbose:
        print('{} recent files (from {} on) found in path {} from bucket {}.'
              .format(len(files), str(from_datetime), path_in_bucket, bucket_name))
    return files


def list_files_from_bucket_path(path_in_bucket, bucket_name, filename_prefix=None, filename_sufix=None,
                                include_datetimes=False):
    if filename_prefix:
        prefix = os.path.join(path_in_bucket, filename_prefix)
    else:
        prefix = path_in_bucket

    s3_client = boto3.client('s3')
    bucket_objects = []
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if response['KeyCount'] > 0:
        while response['IsTruncated']:
            bucket_objects += response['Contents']
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix,
                                                 ContinuationToken=response['NextContinuationToken'])
        bucket_objects += response['Contents']

    if not filename_sufix:
        filename_sufix = ''

    if include_datetimes:
        files = [(obj['Key'], obj['LastModified']) for obj in bucket_objects if obj['Key'].endswith(filename_sufix)]
    else:
        files = [obj['Key'] for obj in bucket_objects if obj['Key'].endswith(filename_sufix)]
    return files


def delete_dir_from_bucket(dir_key, bucket_name):
    print('Deleting objects that start with {} from bucket {}'.format(dir_key, bucket_name))
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.objects.filter(Prefix=dir_key).delete()
    print('Objects deleted')

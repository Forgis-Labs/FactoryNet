#!/usr/bin/env python3
"""
Script to search and download IP-free manuals from Internet Archive.
Downloads PDF files directly to a local directory.
"""

import os
import sys
import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from internetarchive import search_items, get_item, download
from requests.exceptions import HTTPError

def download_manual(
    identifier: str,
    title: str,
    output_dir: str = "manuals",
    retry_count: int = 2
) -> tuple[str, bool, str]:
    """
    Download a manual from Internet Archive by identifier.

    Args:
        identifier (str): The Internet Archive identifier for the manual.
        title (str): The title of the manual.
        output_dir (str, optional): Directory to save the downloaded PDF. Defaults to "manuals".
        retry_count (int, optional): Number of times to retry on transient errors. Defaults to 2.

    Returns:
        tuple[str, bool, str]: (identifier, success, message)
    """
    # Final output path
    final_output_path = os.path.join(output_dir, f"{identifier}.pdf")
    
    # Skip if already downloaded
    if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
        return (identifier, True, f"Already exists: {title[:50]}")
    
    # Retry logic for transient errors
    for attempt in range(retry_count + 1):
        try:
            # Try to download directly - the download function will handle missing PDFs
            # Use the library's download function with glob pattern for PDFs
            download(identifier, destdir=output_dir, glob_pattern='*.pdf', verbose=False)
            
            # The download function creates a subdirectory with the identifier
            # Find the downloaded PDF and move it to the final location
            temp_dir = os.path.join(output_dir, identifier)
            if os.path.exists(temp_dir):
                # Look for PDF files in the temp directory
                for pdf_file in os.listdir(temp_dir):
                    if pdf_file.endswith('.pdf'):
                        temp_file = os.path.join(temp_dir, pdf_file)
                        # Move to final location
                        shutil.move(temp_file, final_output_path)
                        # Remove the now-empty temp directory
                        try:
                            os.rmdir(temp_dir)
                        except OSError:
                            pass
                        break
            
            if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
                return (identifier, True, f"✓ Downloaded: {title[:50]}")
            else:
                return (identifier, False, f"✗ No PDF found: {title[:50]}")
                
        except HTTPError as e:
            # Handle HTTP errors specifically
            if e.response.status_code == 401:
                return (identifier, False, f"✗ Access denied (restricted): {title[:50]}")
            elif e.response.status_code == 403:
                return (identifier, False, f"✗ Forbidden (restricted): {title[:50]}")
            elif e.response.status_code == 404:
                return (identifier, False, f"✗ Not found: {title[:50]}")
            elif e.response.status_code >= 500 and attempt < retry_count:
                # Server error - retry
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                return (identifier, False, f"✗ HTTP {e.response.status_code}: {title[:50]}")
        except Exception as e:
            error_msg = str(e)
            # Check if it's a 401 error in the message
            if "401" in error_msg or "Unauthorized" in error_msg:
                return (identifier, False, f"✗ Access denied (restricted): {title[:50]}")
            elif attempt < retry_count:
                # Retry for other errors
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                return (identifier, False, f"✗ Error: {error_msg[:50]}")
    
    return (identifier, False, f"✗ Failed after {retry_count + 1} attempts: {title[:50]}")

def search_and_download_ip_free_manuals(
    count: int = 100,
    output_dir: str = "manuals",
    max_workers: int = 5
) -> None:
    """
    Search Internet Archive for IP-free manuals and download them.

    Args:
        count (int): Number of manuals to download.
        output_dir (str): Directory to save downloaded files.
        max_workers (int): Maximum number of concurrent download threads.
    """
    print(f"Searching Internet Archive for {count} IP-free manuals...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Query for IP-free machine/equipment/vehicle manuals (public domain or creative commons)
    # Focus strictly on machines, vehicles, equipment, and military vehicles
    # Only include items from 1960 onwards
    query = (
        '(licenseurl:*publicdomain* OR licenseurl:*creativecommons*) AND '
        '(mediatype:texts) AND '
        '(format:"PDF" OR format:"Text PDF") AND '
        '(date:[1960 TO *]) AND '
        '(title:*manual* OR title:*handbook*) AND '
        '(subject:"machinery" OR subject:"machines" OR subject:"equipment" OR '
        'subject:"vehicles" OR subject:"military vehicles" OR subject:"tanks" OR '
        'subject:"trucks" OR subject:"aircraft" OR subject:"automotive" OR '
        'subject:"manufacturing" OR subject:"industrial" OR subject:"technical" OR '
        'subject:"service manual" OR subject:"repair manual" OR subject:"maintenance" OR '
        'subject:"military" OR subject:"army" OR subject:"navy" OR subject:"air force" OR '
        'collection:manuals OR collection:folkscanomy_technical OR '
        'collection:opensource OR collection:governmentpublications)'
    )
    
    # Search with fields we need (including subject and date for better filtering)
    search = search_items(
        query,
        fields=['identifier', 'title', 'licenseurl', 'mediatype', 'subject', 'date']
    )
    
    downloaded = 0
    skipped = 0
    restricted = 0
    
    # First, collect all valid items
    print(f"\nFiltering search results...\n")
    valid_items = []
    
    for item in search:
        if len(valid_items) >= count * 2:  # Collect extra items to account for failures
            break
        
        identifier = item.get('identifier', '')
        title_raw = item.get('title', 'Unknown Title')
        title_lower = title_raw.lower()
        subjects = item.get('subject', [])
        if isinstance(subjects, str):
            subjects = [subjects]
        subjects_lower = ' '.join([s.lower() for s in subjects]) if subjects else ''
        date = item.get('date', '')
        
        if not identifier:
            continue
        
        # Additional date filter: skip if date is before 1960
        if date:
            try:
                year_str = str(date).split('-')[0].split('/')[0]
                year = int(year_str)
                if year < 1960:
                    continue
            except (ValueError, AttributeError):
                pass
        
        # Strict filter: must be machine/vehicle/equipment manual
        manual_keywords = ['manual', 'handbook', 'guide', 'instruction']
        equipment_keywords = [
            'machine', 'machinery', 'equipment', 'vehicle', 'truck', 'tank', 
            'aircraft', 'airplane', 'helicopter', 'automotive', 'motor', 'engine',
            'military', 'army', 'navy', 'air force', 'service', 'repair', 
            'maintenance', 'technical', 'industrial', 'manufacturing', 'tractor',
            'bulldozer', 'excavator', 'crane', 'generator', 'compressor', 'pump',
            'welding', 'lathe', 'mill', 'drill', 'press', 'tool', 'apparatus',
            'system', 'unit', 'component', 'assembly', 'parts', 'operation'
        ]
        
        search_text = f"{title_lower} {subjects_lower}"
        has_manual = any(keyword in search_text for keyword in manual_keywords)
        has_equipment = any(keyword in search_text for keyword in equipment_keywords)
        
        if has_manual and has_equipment:
            valid_items.append((identifier, title_raw))
    
    # Remove duplicates based on identifier
    seen = set()
    unique_items = []
    for identifier, title in valid_items:
        if identifier not in seen:
            seen.add(identifier)
            unique_items.append((identifier, title))
    valid_items = unique_items
    
    print(f"Found {len(valid_items)} unique valid items. Starting parallel downloads...\n")
    
    # Download in parallel using ThreadPoolExecutor
    actual_workers = min(max_workers, count)  # Don't exceed count or specified workers
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all download tasks
        future_to_item = {
            executor.submit(download_manual, identifier, title, output_dir): (identifier, title)
            for identifier, title in valid_items[:count]
        }
        
        # Process completed downloads
        for future in as_completed(future_to_item):
            identifier, title = future_to_item[future]
            try:
                result_id, success, message = future.result()
                if success:
                    downloaded += 1
                    print(f"[{downloaded}/{count}] {message}")
                else:
                    skipped += 1
                    # Track restricted items separately
                    if "restricted" in message.lower() or "forbidden" in message.lower() or "access denied" in message.lower():
                        restricted += 1
                        # Don't print restricted messages - too noisy
                    elif "Already exists" not in message:
                        # Only print other skip messages (errors, no PDF found, etc.)
                        print(f"[Skipped] {message}")
            except Exception as e:
                skipped += 1
                error_msg = str(e)
                if "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg or "Forbidden" in error_msg:
                    restricted += 1
                    # Don't print restricted messages
                else:
                    print(f"[Error] {identifier}: {error_msg[:50]}")
            
            # Progress update every 10 items
            if (downloaded + skipped) % 10 == 0:
                print(f"\nProgress: {downloaded} downloaded, {restricted} restricted, {skipped - restricted} other skipped\n")
            
            if downloaded >= count:
                # Cancel remaining tasks
                for f in future_to_item.keys():
                    f.cancel()
                break
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Restricted/Access denied: {restricted}")
    print(f"  Other skipped/Failed: {skipped - restricted}")
    print(f"  Total processed: {downloaded + skipped}")
    print(f"  Files saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Download IP-free manuals from Internet Archive'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of manuals to download (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='manuals',
        help='Output directory for downloaded files (default: manuals)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel download workers (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.count <= 0:
        print("Error: Count must be greater than 0")
        sys.exit(1)
    
    search_and_download_ip_free_manuals(
        count=args.count,
        output_dir=args.output_dir,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()

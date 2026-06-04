"""
Clean docling-extracted markdown files by removing meaningless character sequences
that result from PDF parsing artifacts.
"""

import re
import os
from pathlib import Path


def is_meaningless_line(line: str) -> bool:
    """
    Determine if a line contains meaningless character sequences.
    
    Args:
        line: The line to check
        
    Returns:
        True if the line should be removed, False otherwise
    """
    # Remove leading/trailing whitespace for analysis
    stripped = line.strip()
    
    # Empty lines are OK
    if not stripped:
        return False
    
    # Check for lines that are mostly /G sequences
    # Pattern: /G followed by digits, repeated many times
    g_pattern = r'/G\d+'
    g_matches = re.findall(g_pattern, stripped)
    
    # If more than 10 /G sequences in a line, it's likely meaningless
    if len(g_matches) > 10:
        return True
    
    # Check if the line is mostly /G sequences (>70% of non-whitespace content)
    if g_matches:
        g_content_length = sum(len(match) for match in g_matches)
        total_length = len(stripped.replace(' ', ''))
        if total_length > 0 and (g_content_length / total_length) > 0.7:
            return True
    
    # Check for other repeated meaningless patterns
    # Pattern: sequences like /c1, /c2, etc. repeated
    c_pattern = r'/c\d+'
    c_matches = re.findall(c_pattern, stripped)
    if len(c_matches) > 5:
        return True
    
    return False


def is_meaningless_table_row(line: str) -> bool:
    """
    Check if a table row is filled with meaningless sequences.
    
    Args:
        line: The line to check
        
    Returns:
        True if it's a meaningless table row
    """
    # Check if it's a table row (contains |)
    if '|' not in line:
        return False
    
    # Split by | and check each cell
    cells = line.split('|')
    
    # Count cells with /G sequences
    meaningless_cells = 0
    total_cells = 0
    
    for cell in cells:
        cell_stripped = cell.strip()
        if not cell_stripped:
            continue
        total_cells += 1
        
        # Check if cell is mostly /G sequences
        g_matches = re.findall(r'/G\d+', cell_stripped)
        if len(g_matches) > 3:
            meaningless_cells += 1
    
    # If most cells are meaningless, remove the row
    if total_cells > 0 and (meaningless_cells / total_cells) > 0.5:
        return True
    
    return False


def clean_markdown_content(content: str) -> str:
    """
    Clean markdown content by removing meaningless sequences.
    
    Args:
        content: The markdown content to clean
        
    Returns:
        Cleaned markdown content
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    in_table = False
    skip_next_separator = False
    
    for i, line in enumerate(lines):
        # Check if we're entering/leaving a table
        if '|' in line and i + 1 < len(lines) and re.match(r'\s*\|[\s\-:|]+\|', lines[i + 1]):
            in_table = True
        
        # Skip meaningless lines
        if is_meaningless_line(line):
            continue
        
        # Skip meaningless table rows
        if is_meaningless_table_row(line):
            # If this is a header row, also skip the separator
            if i + 1 < len(lines) and re.match(r'\s*\|[\s\-:|]+\|', lines[i + 1]):
                skip_next_separator = True
            continue
        
        # Skip separator if previous header was meaningless
        if skip_next_separator and re.match(r'\s*\|[\s\-:|]+\|', line):
            skip_next_separator = False
            continue
        
        # Keep the line
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_file(input_path: Path, output_path: Path):
    """
    Clean a single markdown file.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
    """
    print(f"Cleaning {input_path.name}...")
    
    # Read the file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the content
    cleaned_content = clean_markdown_content(content)
    
    # Calculate reduction
    original_lines = len(content.split('\n'))
    cleaned_lines = len(cleaned_content.split('\n'))
    removed_lines = original_lines - cleaned_lines
    
    print(f"  Original: {original_lines} lines")
    print(f"  Cleaned: {cleaned_lines} lines")
    print(f"  Removed: {removed_lines} lines ({removed_lines/original_lines*100:.1f}%)")
    
    # Write the cleaned content
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"  Saved to {output_path}")


def main():
    """Main function to clean all markdown files."""
    # Define paths
    input_dir = Path(r"C:\Users\Jonaspetersen\.gemini\antigravity\scratch\FactoryNet\data\docling_outputs")
    output_dir = Path(r"C:\Users\Jonaspetersen\.gemini\antigravity\scratch\FactoryNet\data\docling_outputs_cleaned")
    
    # Find all .md files
    md_files = list(input_dir.glob("*.md"))
    
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return
    
    print(f"Found {len(md_files)} markdown file(s) to clean\n")
    
    # Clean each file
    for md_file in md_files:
        output_path = output_dir / md_file.name
        clean_file(md_file, output_path)
        print()
    
    print("Cleaning complete!")


if __name__ == "__main__":
    main()

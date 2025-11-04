#!/bin/bash

# Document Scanner API Test Script

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if URL is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: API URL is required${NC}"
    echo "Usage: ./test_api.sh <API_URL>"
    echo "Example: ./test_api.sh https://document-scanner-xxxxx.run.app"
    exit 1
fi

API_URL="$1"

echo -e "${YELLOW}Document Scanner API Test${NC}"
echo "API URL: $API_URL"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}[1/4] Testing Health Check...${NC}"
response=$(curl -s "${API_URL}/health")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "Response: $response"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi
echo ""

# Test 2: Root Endpoint
echo -e "${YELLOW}[2/4] Testing Root Endpoint...${NC}"
response=$(curl -s "${API_URL}/")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Root endpoint passed${NC}"
    echo "Response: $response"
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
fi
echo ""

# Test 3: Check if test image exists
echo -e "${YELLOW}[3/4] Checking for test image...${NC}"
if [ ! -f "test_document.jpg" ] && [ ! -f "test_document.png" ]; then
    echo -e "${YELLOW}! No test image found. Please provide a test_document.jpg or test_document.png${NC}"
    echo -e "${YELLOW}  You can create one using:${NC}"
    echo "  - Take a photo of a document"
    echo "  - Or download a sample from the internet"
    echo ""
    echo -e "${YELLOW}Skipping scan test...${NC}"
else
    # Find the test image
    if [ -f "test_document.jpg" ]; then
        TEST_IMAGE="test_document.jpg"
    else
        TEST_IMAGE="test_document.png"
    fi
    
    echo -e "${GREEN}✓ Found test image: $TEST_IMAGE${NC}"
    echo ""
    
    # Test 4: Scan endpoint
    echo -e "${YELLOW}[4/4] Testing Scan Endpoint...${NC}"
    echo "Uploading image and generating PDF..."
    
    curl -X POST "${API_URL}/scan" \
        -F "file=@${TEST_IMAGE}" \
        -F "enhance=true" \
        -F "page_size=A4" \
        -o "output_scanned.pdf" \
        -w "\nHTTP Status: %{http_code}\n"
    
    if [ $? -eq 0 ] && [ -f "output_scanned.pdf" ]; then
        file_size=$(wc -c < "output_scanned.pdf")
        if [ $file_size -gt 1000 ]; then
            echo -e "${GREEN}✓ Scan test passed${NC}"
            echo "PDF generated: output_scanned.pdf (${file_size} bytes)"
        else
            echo -e "${RED}✗ Scan test failed - PDF too small${NC}"
            cat output_scanned.pdf
        fi
    else
        echo -e "${RED}✗ Scan test failed${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Test completed!${NC}"

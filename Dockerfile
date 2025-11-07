# Bước 1: Sử dụng image Python 3.10
FROM python:3.10-slim

# Đặt biến môi trường để Selenium biết không cần tìm driver
ENV PATH /usr/bin:$PATH

# Bước 2: Cài đặt các gói hệ thống và Google Chrome
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    # Cài đặt Google Chrome
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    # Dọn dẹp apt
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Bước 3: Cài đặt ChromeDriver
# Tìm phiên bản Chrome đã cài
RUN CHROME_VERSION=$(google-chrome --version | cut -d " " -f 3 | cut -d "." -f 1-3) \
    # Tìm phiên bản ChromeDriver tương ứng
    && CHROME_DRIVER_VERSION=$(wget -qO- "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION}") \
    && echo "Đang cài ChromeDriver phiên bản: $CHROME_DRIVER_VERSION" \
    && wget -q "https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip \
    && mv chromedriver /usr/bin/chromedriver \
    && chown root:root /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    # Dọn dẹp file zip
    && rm chromedriver_linux64.zip

# Bước 4: Cài đặt Python
WORKDIR /app

# Sao chép file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Bước 5: Sao chép code ứng dụng
COPY base_hiring_api.py .

# Bước 6: Lệnh chạy ứng dụng
# Render sẽ dùng port 10000 để kết nối
CMD ["uvicorn", "base_hiring_api:app", "--host", "0.0.0.0", "--port", "10000"]

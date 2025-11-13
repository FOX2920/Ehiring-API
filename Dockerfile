# Sử dụng base image Python 3.10
FROM python:3.10-slim

# Đặt biến môi trường PORT mà Cloud Run yêu cầu
# Ứng dụng của bạn phải lắng nghe trên port này
ENV PORT 8080

# Đặt thư mục làm việc
WORKDIR /app

# Copy file requirements
COPY requirements.txt ./

# Cài đặt dependencies
# --no-cache-dir giúp giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

# Copy file app.py vào image
COPY app.py .

# Lệnh để chạy ứng dụng của bạn
# Chúng ta dùng gunicorn (có trong requirements.txt)
# Nó sẽ tự động dùng $PORT
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:$PORT", "app:app"]

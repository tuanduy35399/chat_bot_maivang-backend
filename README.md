# Chatbot chẩn đoán bệnh cây mai vàng (Baseline)

Mục tiêu
Xây dựng chatbot cơ bản có thể:

- Nhận mô tả triệu chứng từ người dùng
- Gợi ý loại bệnh và cách xử lý
- Cho phép tải ảnh minh họa (chưa có AI nhận diện)
- Là nền tảng cho việc phát triển mô hình chẩn đoán bằng hình ảnh sau này

## Cách cài và chạy
## Cần chạy trong môi trường venv
### Chạy lệnh cài môi trường (nếu chưa có)
python -m venv .env
source .env/bin/activate

### Cài đặt thư viện cần thiết

pip install streamlit pillow

## Khôi phục mật khẩu bằng email

Chạy `python manage.py migrate` sau khi cập nhật mã nguồn. Cấu hình các biến môi trường trên máy chủ (không đưa mật khẩu vào Git):

```dotenv
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=true
EMAIL_HOST_USER=your-sender@gmail.com
EMAIL_HOST_PASSWORD=your-gmail-app-password
DEFAULT_FROM_EMAIL=your-sender@gmail.com
```

Với Gmail, bật xác minh hai bước và tạo App Password cho tài khoản gửi. Email người nhận là email đã đăng ký của tài khoản, không bắt buộc thuộc Gmail. Trong môi trường phát triển có thể dùng `EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend` để xem email trong console thay vì gửi thật.

- `POST /api/v1/user/password-reset/request/`: `{ "email": "user@gmail.com" }` yêu cầu OTP; phản hồi thông thường không xác nhận email có tồn tại. Mỗi tài khoản được gửi lại sau 60 giây.
- `POST /api/v1/user/password-reset/confirm/`: `{ "email": "user@gmail.com", "otp": "123456", "new_password": "..." }` đổi mật khẩu. Mã có hiệu lực 10 phút, tối đa 5 lần nhập sai và chỉ dùng một lần. Mật khẩu phải đáp ứng bộ kiểm tra Django.

Giới hạn yêu cầu theo IP dùng cache Django; khi triển khai nhiều tiến trình/máy chủ cần cấu hình cache dùng chung và proxy chuyển tiếp IP đáng tin cậy. Việc đổi mật khẩu chưa thu hồi JWT đã cấp trước đó; token cũ vẫn có hiệu lực đến khi hết hạn theo cấu hình JWT hiện tại.

# SignMeet

## Giới thiệu
Ứng dụng video call hỗ trợ dịch phụ đề đa ngôn ngữ và ngôn ngữ ký hiệu dành cho người khiếm thính và người có nhu cầu giao tiếp với người khiếm thính. Ứng dụng cung cấp các chức năng cơ bản như:
  
## Giao diện
### Trang chủ
![image](https://github.com/user-attachments/assets/cd06ec52-c1e1-4be1-9d27-f647f3836c1c)



### Trang Workplace
![{F0822BE1-2CAB-4547-A692-CF46E63DF2A3}](https://github.com/user-attachments/assets/1a57e63e-8fc1-453a-bca3-1573630a7e13)

### Trang đặt lịch họp
![{980BFD51-178E-4440-8ED0-02958605B97A}](https://github.com/user-attachments/assets/84e04633-0e5b-4d37-ac6a-d4ef760c3c21)

## Phòng họp
![{286DEA08-E85F-464C-9819-55B83B44215A}](https://github.com/user-attachments/assets/bb1f5275-5050-4293-b87e-36e62408b9e1)

## Thành phần cụ thể
### Gọi video
Ứng dụng sử dụng giao thức WebRTC để truyền tải âm thanh và video giữa các người dùng với nhau
### Nhắn tin
Dùng firebase database realtime để lưu trữ và chuyển tin nhắn
### Speech-to-Text đa ngôn ngữ:
![{08C37C6B-D71F-4644-B961-158EFC1806C7}](https://github.com/user-attachments/assets/b6158b1c-de3a-4114-a624-88d3e11fec9c)
### Điều khiển bằng giọng nói
TTích hợp WebNavigation vào SignMeet nhằm bổ sung khả năng điều hướng web bằng giọng nói và cử chỉ, tận dụng Whisper để nâng cao hiệu suất nhận diện giọng nói. Phần này tập trung chi tiết vào phương án tích hợp, bao gồm kiến trúc hệ thống và các bước triển khai

## Công nghệ:
Firebase
Golang
NextJS
Google Speech-To-Text
Kafka
Redis
Pytorch
...

## Run
```
cd /frontend
npm install
npm run dev

cd /backend
add API KEY to env file
node speechToText-v2/js/server.js
```


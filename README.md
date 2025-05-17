Link video demo:
https://www.youtube.com/watch?v=vKxh4vjGzHI

Link slide demo:

https://www.canva.com/design/DAGni0Li6fU/K-1VTWUkTA82XfuLhu80VA/edit?fbclid=IwZXh0bgNhZW0CMTEAAR4RPsWJ4d-zxvrluDpWt5_hByf_LZgch_LGbDgC-CBqOzvf58PD-XzkzwSnbA_aem_OGjBYSZE75iNGme3qjiQgg


# Cờ Vua

Ứng dụng trò chơi cờ vua với giao diện đồ họa sử dụng **Pygame** (Python) kết hợp engine cờ tự viết bằng **C++** theo giao thức UCI.

---

## 🔎 Mô tả dự án

- **Giao diện**: Bàn cờ 8×8, thanh bên (history nước đi, nút New Game/Undo/Redo/Home/Exit).
- **Chế độ chơi**:  
  - Hai người chơi (hotseat).  
  - Chơi với AI (gọi engine C++ qua UCI).
- **Tính năng chính**:  
  - Undo/Redo nước đi.  
  - Phong hậu (promotion) khi đến cuối bàn.  
  - Hiệu ứng âm thanh khi di chuyển, ăn quân, kết thúc ván.  
  - Highlight khi chiếu, thông báo chiếu hết/hoà.  
  - Tùy chỉnh thời gian suy nghĩ của AI

---

## ⚙️ Yêu cầu & Cài đặt

1. **Yêu cầu hệ thống**  
   - Python ≥ 3.8  
   - Trình biên dịch C++ (g++/clang++ hỗ trợ C++17)  
   - Windows / Linux (có thể cần điều chỉnh `ENGINE_PATH`)

2. **Cài đặt engine C++**  
   ```bash
   cd engine
   make

## Đóng góp thành viên:
- khánh, huy anh: backend + engine
- hào, bảo: frontend + logic game + hỗ trợ backend và engine

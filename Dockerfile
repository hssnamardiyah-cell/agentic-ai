FROM python:3.9-slim

# 1. Install dependencies sistem
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade PIP agar lebih stabil
RUN pip install --upgrade pip

# 3. TRIK PENYELAMAT: Install dependency Torch dari server standar DULUAN
# Ini mencegah Docker mengambil file "rusak" dari server PyTorch
RUN pip install "typing-extensions>=4.10.0" sympy networkx jinja2 fsspec filelock

# 4. Setelah aman, baru install PyTorch CPU versi terbaru
RUN pip install --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

# 5. Copy requirements dan install sisanya
COPY requirements.txt .
# Pastikan baris 'torch' SUDAH DIHAPUS dari requirements.txt Anda
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy semua file aplikasi
COPY . .

# 7. Konfigurasi Port
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
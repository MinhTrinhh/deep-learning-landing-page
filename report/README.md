- Lệnh chạy compile latex report
cd report\report-assignment-1
latexmk -pdf report.tex

==>> PDF và các file tạm được tạo trong thư mục build.

- Các lệnh khác: 
Dọn các file tạm nhưng giữ lại PDF:
latexmk -c report.tex

Xóa toàn bộ output, gồm cả PDF:
latexmk -C report.tex


- Tùy IDE, cần cấu hình LaTeX hoặc LaTeX Workshop để output được ghi vào thư mục build thay vì thư mục chứa report.tex.

+ Ví dụ với VS Code:
"latex-workshop.latex.outDir": "%DIR%/build"

+ Nếu muốn dùng XeLaTeX:
latexmk -xelatex report.tex

Cấu hình tương ứng trong latexmkrc:
# $xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
# $pdf_mode = 5;
# $out_dir = 'build';

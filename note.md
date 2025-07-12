# Các tạo project Django
+ Tạo folder để chứa project
+ Mở cmd: dijango-admin startproject <Tên project>

# câu lệnh chạy server ảo:
+ python manage.py runserver (dành cho windows)
+ Mở trình duyệt nhập localhost:8000
+ Muốn run cổng khác thì: python manage.py runserver 8080

# tạo một WebApp trong Django
+ WebApp là 1 trangweb đó
+ gõ cmd: python manage.py startapp <tên app>
+ sao khi đã tạo app, vào PythonWeb/settings.py thêm app mới khởi chạy
    vào INSTALLED_APPS, ví dụ app tên home thì thêm 'home' vaò
+ xong rồi thì migrate, vào CMD gõ: python manage.py migrate
+ Tạo các hàm để xử lý v,v
+ Trong app tạo một urls.py, thêm partterns v.v
+ trong urls của PythonWeb, import thêm include để gọi urls của app home

## Kiểm thử phần mềm project:
    + tạo ra các test case trong file tests.py
    + xem thêm tỏng file tests.py của home
    + Viết hàm kiểm tra xong thì có thể check bằng cách cmd = python manage.py test home 
    + Nếu muốn test cả project  thì python manage.py test

## Tạo tempales:
    + Trong webapp, tạo 1 folder templates/pages/base.html

## Tạo bảng (database) và sử dụng trong models:
    + Trong webapp, vào models, tạo một class -> class là 1 bảng, code xong thì
    + cmd: python manage.py makemigrations <tên webapp> => lệnh này để thông báo có chỉnh sửa, thêm class trong model
    + vào trong migrations check, có id tự tăng
    + Cập nhật lại csdl sqlite3: python manage.py migrate

## tương tác với database bằng python
    + cmd: 
    + Ví dụ:
        - Import dữ liệu vào một bảng (class): 
            + from <tên webapp>.models import <tên bảng>
            + a = <tên bảng>() => ví dụ a = Post()
            + a.<thuộc tính bảng> = '' 
            + a.save()
        - câu lệnh xem tất cả bảng => <Bảng>.objects.all()
        - Nhưng lúc này nó sẽ hiện 1 cột là id thôi
        - Vào trong class đang xét, thêm hàm override phương thức str chỗ đó :
                def __str__(self):
                 return self.title => (hoặc một thuộc tính nào khác)
    => Vậy là nó sẽ hiện như select title from Post
    * Nếu có check trong DB SQLite thì sau khi xem xong, DB Sqlite mới thêm được dữ liệu mới trong shell
## Các lưu ý:
- file settings.py là file chỉnh sửa cấu hình project (database v.v)
- file urls.py là file đặt tên đường dẫn, liên kết các tương tác
- xử lý các respone  của người dùng, trong các app thì sẽ code trong views.py, tạo các hàm def để xử lý
- pip install Pillow
## thao tác html:
File > Preferences > Settings.
Ở góc trên bên phải của Settings, bạn sẽ thấy biểu tượng hình tập tin (có thể hiển thị là “Open Settings JSON”).
Nhấn vào đó để mở file settings.json.
Trong settings.json, thêm đoạn sau để Emmet hỗ trợ trong file Django HTML:
"emmet.includeLanguages": {
    "django-html": "html"
}

Links xem cấu hình boostrap cho dự án:
https://www.howkteam.com/course/lap-trinh-web-voi-python-bang-django/file-tinh-va-thiet-ke-web-bang-bootstrap-trong-python-django-1519


## Hệ thống admin có sẵn trong Django
 - Giải quyết vấn đề thêm sửa xóa dữ liệu CURD sẵn
 - Tạo tài khoản admin: python manage.py createsuperuser
    Tên: admin, mail: huongtruong290104@gmail.com
    mật khẩu: 123
- runserver vào /admin là được.
- Làm sao để quản lý dữ liệu các bảng mình tạo sau?
    + vào admin.py (trong webapp chứa class đó): import .models import <tên class>
        admin.site.register(<tên class>)
- Chỉnh sửa hiển thị
    + Tạo một hàm override hàm Admin:
            class PostAdmin(admin.ModelAdmin):
                list_display = ['title', 'date']
                list_filter = ['date']  ## filer theo trường date, ra sẵn theo ngày, ngày bất kì hôm nay, tháng, năm v,v

            admin.site.register(Post, PostAdmin)
- Hệ thống hiện tại trông hơi nham nhở, ta có thể dùng các thư viện:
✅ 4. Dùng thư viện bên ngoài để "đẹp" hơn
        Bạn có thể dùng các giao diện admin tùy chỉnh như:

        django-admin-interface

        django-grappelli

        jet-django

        Cài bằng pip, thêm vào INSTALLED_APPS, và cấu hình theo hướng dẫn của từng thư viện.

## Cách hiển thị liệt kê dữ liệu lên webapp
    - vào views.py trong webapp đang xét
    - import Class (bảng cần dùng)
    - Tạo hàm để hiển thị liệt kê:
    def list(request):
   Data = {'Post':Post.objects.all().order_by("-date")  } 
   return render(request, 'blog/blog.html', Data)

## Hiển thị trang theo id:
     - trong urls webapp đang xét, thêm path:

## sử dụng TestCase:
    - nó sẽ lấy database ảo nên yên tâm
    class BlogTests(TestCase):
    ## overide
    def setUp(self):
        Post.objects.create(
            title='myTitle',
            body = 'just a Test'
        ) ## test nên sẽ là database ảo 
    
    def test_string_representation(self):
        post = Post(title='My hi Title')
        self.assertEqual(str(post), post.title)
    
    def test_post_list_view(self):
        response= self.client.get('/blog/')
        self.assertEqual(response.status_code, 200) ## xem coi có bằng 200 không
        self.assertContains(response, 'myTitle')
        self.assertTemplateUsed(response, 'blog/blog.html')
    
    def test_post_details_view(self):
        response= self.client.get('/blog/1/')
        self.assertEqual(response.status_code, 200) ## xem coi có bằng 200 không
        self.assertContains(response, 'myTitle')
        self.assertTemplateUsed(response, 'blog/post.html')
    
## loại bỏ hardcode url: đặt tên
    - {% url 'post' post.id %}
    - {% url 'blog' %}



## xử lý lỗi 404
    - viết trong views home đi

## map đường links để lưu ảnh, video v.v
    - settings.py:
        MEDIA_URL = '/media/'   => theo tên folder
        MEDIA_ROOT = os.path.join(BASE_DIR,'media')
    - vào urls  tổng khai báo:
    if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

## Tạo form đăng ký tài khoản
    - tạo forms.py trong webapp đang xet
    - Cho nó urls
    - Trong froms viết class khởi tạo class đăng ký, gọi class đó trong views (hàm resiter các kiểu)


## login / logout trong django:
    - from django.contrib.auth import views as ath_views

## genericView

## thêm bình luận
 
## Thêm sửa xóa
    
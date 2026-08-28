from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from .models import Profile

class UserModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='johndoe',
            email='john@example.com',
            password='secretpassword123'
        )
        self.profile = Profile.objects.create(
            user=self.user,
            bio='AI & Computer Vision Researcher'
        )

    def test_profile_creation(self):
        self.assertEqual(self.profile.user.username, 'johndoe')
        self.assertEqual(self.profile.bio, 'AI & Computer Vision Researcher')
        self.assertEqual(self.profile.total_detections, 0)
        self.assertIn("johndoe's Profile", str(self.profile))


class UserAuthViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='janedoe',
            email='jane@example.com',
            password='securepassword123'
        )
        self.profile = Profile.objects.create(user=self.user)

    def test_login_page_renders(self):
        response = self.client.get(reverse('users:login'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/login.html')

    def test_register_page_renders(self):
        response = self.client.get(reverse('users:register'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/register.html')

    def test_successful_login(self):
        response = self.client.post(reverse('users:login'), {
            'username': 'janedoe',
            'password': 'securepassword123'
        })
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('dashboard:dashboard'))

    def test_profile_page_requires_login(self):
        response = self.client.get(reverse('users:profile'))
        self.assertEqual(response.status_code, 302)

    def test_profile_page_logged_in(self):
        self.client.login(username='janedoe', password='securepassword123')
        response = self.client.get(reverse('users:profile'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/profile.html')

    def test_password_reset_views_render(self):
        response = self.client.get(reverse('users:password_reset'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/password_reset.html')

        response = self.client.get(reverse('users:password_reset_done'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/password_reset_done.html')

        response = self.client.get(reverse('users:password_reset_complete'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'users/password_reset_complete.html')

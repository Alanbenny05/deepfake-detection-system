from django import forms

class MediaUploadForm(forms.Form):
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
    ]
    
    media_type = forms.ChoiceField(choices=MEDIA_TYPES, widget=forms.RadioSelect)
    media_file = forms.FileField(
        label='Upload File',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

class YouTubeURLForm(forms.Form):
    url = forms.URLField(
        label='YouTube URL',
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://www.youtube.com/watch?v=...'
        })
    )

class InstagramURLForm(forms.Form):
    url = forms.URLField(
        label='Instagram Reel URL',
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://www.instagram.com/reel/...'
        })
    )
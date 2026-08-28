"""
Migration 0002: Add multi-signal detection fields to Detection model.

All new fields are nullable/have defaults so existing records are
unaffected and the migration is fully backward compatible.
"""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0001_initial'),
    ]

    operations = [
        # Update result field choices to include 'uncertain'
        migrations.AlterField(
            model_name='detection',
            name='result',
            field=models.CharField(
                choices=[
                    ('real',      'Real'),
                    ('fake',      'AI Generated'),
                    ('uncertain', 'Uncertain'),
                ],
                default='uncertain',
                max_length=20,
            ),
        ),

        # Update model_version default
        migrations.AlterField(
            model_name='detection',
            name='model_version',
            field=models.CharField(default='v2.0', max_length=50),
        ),

        # ── New multi-signal fields ─────────────────────────────────────────
        migrations.AddField(
            model_name='detection',
            name='cnn_score',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Raw CNN fake probability [0-1]',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='forensic_score',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Forensic analysis fake probability [0-1]',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='provenance_status',
            field=models.CharField(
                blank=True, default='unavailable', max_length=20,
                help_text='detected | not_detected | unavailable | error',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='provenance_markers',
            field=models.TextField(
                blank=True, default='',
                help_text='JSON list of found provenance markers',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='frames_analyzed',
            field=models.IntegerField(
                blank=True, null=True,
                help_text='Number of video frames analysed',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='suspicious_frames',
            field=models.IntegerField(
                blank=True, null=True,
                help_text='Number of frames flagged as suspicious',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='fake_frame_pct',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Percentage of face frames that appear fake',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='temporal_consistency',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Temporal consistency score [0-1] for video',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='final_score',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Ensemble final fake probability [0-1]',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='classification',
            field=models.CharField(
                blank=True, default='UNCERTAIN', max_length=25,
                choices=[
                    ('HIGH_CONFIDENCE_REAL', 'High Confidence Real'),
                    ('LIKELY_REAL',          'Likely Real'),
                    ('UNCERTAIN',            'Uncertain'),
                    ('LIKELY_FAKE',          'Likely Fake'),
                    ('HIGH_CONFIDENCE_FAKE', 'High Confidence Fake'),
                ],
                help_text='Ensemble classification tier',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='explanation',
            field=models.TextField(
                blank=True, default='',
                help_text='Human-readable explanation of the result',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='processing_time',
            field=models.FloatField(
                blank=True, null=True,
                help_text='Total detection processing time in seconds',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='source_type',
            field=models.CharField(
                blank=True, default='upload', max_length=20,
                help_text='upload | youtube | instagram',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='forensic_flags',
            field=models.TextField(
                blank=True, default='',
                help_text='JSON list of forensic warning flags',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='face_found',
            field=models.BooleanField(
                blank=True, null=True,
                help_text='Whether a face was detected in the media',
            ),
        ),
        migrations.AddField(
            model_name='detection',
            name='cnn_mode',
            field=models.CharField(
                blank=True, default='', max_length=20,
                help_text='cnn | heuristic | unavailable',
            ),
        ),
    ]

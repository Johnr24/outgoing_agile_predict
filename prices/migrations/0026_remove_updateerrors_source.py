# Generated by Django 4.2.11 on 2025-02-06 16:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0025_updateerrors"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="updateerrors",
            name="source",
        ),
    ]

import django.db.models.deletion
from django.db import migrations, models


def backfill_reply_to(apps, schema_editor):
    ChatMessage = apps.get_model("history", "ChatMessage")
    history_ids = (
        ChatMessage.objects.values_list("history_id", flat=True)
        .distinct()
    )
    to_update = []
    for hid in history_ids:
        messages = list(
            ChatMessage.objects.filter(history_id=hid).order_by("created_at", "id")
        )
        n = len(messages)
        for i in range(n - 1):
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                prev_is_user = i > 0 and messages[i - 1].role == "user"
                next_is_assistant = i + 2 < n and messages[i + 2].role == "assistant"
                if not prev_is_user and not next_is_assistant and messages[i + 1].reply_to_id is None:
                    messages[i + 1].reply_to_id = messages[i].id
                    to_update.append(messages[i + 1])
    if to_update:
        ChatMessage.objects.bulk_update(to_update, ["reply_to"])


class Migration(migrations.Migration):

    dependencies = [
        ("history", "0001_initial"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="chatmessage",
            options={"ordering": ["created_at", "id"]},
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="has_image",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="detections",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="processing_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("complete", "Complete"),
                    ("failed", "Failed"),
                ],
                default="complete",
                max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="reply_to",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="assistant_replies",
                to="history.chatmessage",
            ),
        ),
        migrations.AlterField(
            model_name="chatmessage",
            name="content",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.RunPython(
            backfill_reply_to,
            reverse_code=migrations.RunPython.noop,
        ),
    ]

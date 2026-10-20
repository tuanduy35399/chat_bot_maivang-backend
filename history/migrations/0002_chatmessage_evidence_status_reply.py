import django.db.models.deletion
from django.db import migrations, models


class HistoryWindow:
    def __init__(self, commit_callback):
        self.commit_callback = commit_callback
        self.reset()

    def reset(self):
        self.prev_role = None
        self.candidate_user = None
        self.candidate_pair = None

    def process(self, msg):
        role = msg.role
        if self.candidate_pair is not None:
            if role == "assistant":
                self.candidate_pair = None
            else:
                cand_u, cand_a = self.candidate_pair
                self.candidate_pair = None
                self.commit_callback(cand_u, cand_a)

        if role == "user":
            if self.prev_role == "user":
                self.candidate_user = None
            else:
                self.candidate_user = msg
        elif role == "assistant":
            if self.candidate_user is not None:
                if msg.reply_to_id is None:
                    self.candidate_pair = (self.candidate_user, msg)
                self.candidate_user = None

        self.prev_role = role

    def finish(self):
        if self.candidate_pair is not None:
            cand_u, cand_a = self.candidate_pair
            self.candidate_pair = None
            self.commit_callback(cand_u, cand_a)
        self.reset()


def backfill_reply_to(apps, schema_editor):
    ChatMessage = apps.get_model("history", "ChatMessage")
    qs = ChatMessage.objects.only(
        "id", "history_id", "role", "reply_to_id", "created_at"
    ).order_by("history_id", "created_at", "id")

    to_update = []

    def commit_pair(user_msg, assistant_msg):
        assistant_msg.reply_to_id = user_msg.id
        to_update.append(assistant_msg)
        if len(to_update) >= 500:
            ChatMessage.objects.bulk_update(to_update, ["reply_to"], batch_size=500)
            to_update.clear()

    window = HistoryWindow(commit_pair)
    current_history_id = None

    for msg in qs.iterator(chunk_size=1000):
        if msg.history_id != current_history_id:
            if current_history_id is not None:
                window.finish()
            current_history_id = msg.history_id
        window.process(msg)

    if current_history_id is not None:
        window.finish()

    if to_update:
        ChatMessage.objects.bulk_update(to_update, ["reply_to"], batch_size=500)
        to_update.clear()


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

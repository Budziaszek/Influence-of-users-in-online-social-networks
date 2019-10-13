from enum import Enum


class Mode(Enum):
    comments_to_posts_and_comments = "comments_to_posts_and_comments"
    comments_to_posts = "comments_to_posts"
    comments_to_comments = "comments_to_comment"

    comments_to_posts_and_comments_from_others = "comments_to_posts_and_comments_from_others"
    comments_to_posts_from_others = "comments_to_posts_from_others"
    comments_to_comments_from_others = "comments_to_comment_from_others"

    def __init__(self, *args):
        super().__init__()
        if self.value is "comments_to_posts_and_comments_from_others":
            self.do_read_comments_to_comments = False
            self.do_read_comments_to_posts = False
            self.do_read_comments_to_comments_from_others = True
            self.do_read_comments_to_posts_from_others = True
        elif self.value is "comments_to_posts_from_others":
            self.do_read_comments_to_comments = False
            self.do_read_comments_to_posts = False
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = True
        elif self.value is "comments_to_comment_from_others":
            self.do_read_comments_to_comments = False
            self.do_read_comments_to_posts = False
            self.do_read_comments_to_comments_from_others = True
            self.do_read_comments_to_posts_from_others = False
        if self.value is "comments_to_posts_and_comments":
            self.do_read_comments_to_comments = True
            self.do_read_comments_to_posts = True
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False
        elif self.value is "comments_to_posts":
            self.do_read_comments_to_comments = False
            self.do_read_comments_to_posts = True
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False
        elif self.value is "comments_to_comment":
            self.do_read_comments_to_comments = True
            self.do_read_comments_to_posts = False
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False




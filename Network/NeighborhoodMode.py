from enum import Enum


class NeighborhoodMode(Enum):
    COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS = "comments_to_posts_and_comments_from_others"
    COMMENTS_TO_POSTS_FROM_OTHERS = "comments_to_posts_from_others"
    COMMENTS_TO_COMMENTS_FROM_OTHERS = "comments_to_comment_from_others"

    COMMENTS_TO_POSTS_AND_COMMENTS = "comments_to_posts_and_comments"
    COMMENTS_TO_POSTS = "comments_to_posts"
    COMMENTS_TO_COMMENTS = "comments_to_comment"

    def short(self):
        if self.value is "comments_to_posts_and_comments_from_others":
            return "PCO"
        if self.value is "comments_to_posts_from_others":
            return "PO"
        if self.value is "comments_to_comment_from_others":
            return "CO"
        if self.value is "comments_to_posts_and_comments":
            return "PC"
        if self.value is "comments_to_posts":
            return "P"
        if self.value is "comments_to_comment":
            return "C"

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
        elif self.value is "comments_to_posts_and_comments":
            self.do_read_comments_to_comments = True
            self.do_read_comments_to_posts = True
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False
        elif self.value is "comments_to_posts":
            self.do_read_comments_to_comments = False
            self.do_read_comments_to_posts = True
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False
        elif self.value is "comments_to_comments":
            self.do_read_comments_to_comments = True
            self.do_read_comments_to_posts = False
            self.do_read_comments_to_comments_from_others = False
            self.do_read_comments_to_posts_from_others = False




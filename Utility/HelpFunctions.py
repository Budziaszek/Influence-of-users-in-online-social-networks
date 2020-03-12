class HelpFunctions:

    @staticmethod
    def without_zeros(value):
        if value == 0:
            return False
        return True

    @staticmethod
    def without_none(value):
        if value is None:
            return False
        return True


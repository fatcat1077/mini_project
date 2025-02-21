class animal:
    def __init__(self,name):
        self.name=name
    def make_sound(self):
        print("make sound",self.name)
class dog(animal):
    def make_sound(self):
        return super().make_sound()
class cat(animal):
    def make_sound(self):
        return super().make_sound()
cat1=cat("kiki")
dog1=dog("kigy")
cat1.make_sound()
dog1.make_sound()

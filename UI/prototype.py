from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout 

class TestInterface(BoxLayout):
    def __init__(self):
        super(TestInterface, self).__init__()
        self.orientation = 'vertical'
        btn = Button(text='Click me!')
        self.add_widget(btn)

class TestApp(App):
    def build(self):
        return TestInterface()

if __name__ == '__main__':
    TestApp().run()
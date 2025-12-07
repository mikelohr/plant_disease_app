from kivy.app import App
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from classifier import LeafClassifier

class LeafApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.classifier = LeafClassifier(r"C:\Users\mlohr\DeepLearning\anaconda3\plant_disease_app\model\mobilenet_model.tflite", r"C:\Users\mlohr\DeepLearning\anaconda3\plant_disease_app\model\labels.txt")
        self.image = KivyImage()
        self.label = Label(text="Upload a leaf image to diagnose")
        self.button = Button(text="Choose Image", on_press=self.choose_image)
        self.add_widget(self.image)
        self.add_widget(self.label)
        self.add_widget(self.button)
        
    def choose_image(self, instance):
        layout = BoxLayout(orientation='vertical')
        chooser = FileChooserIconView(filters=["*.jpg", "*.jpeg", "*.png"])
        confirm_btn = Button(text="Select")
        
        layout.add_widget(chooser)
        layout.add_widget(confirm_btn)
        
        popup = Popup(title="Select Leaf Image", content=layout, size_hint=(0.9, 0.9))
    
        confirm_btn.bind(on_release=lambda _: self.on_file_selected(chooser.selection, popup))
        popup.open()

    def on_file_selected(self, selection, popup):
        if selection:
            path = selection[0]
            self.classify_image(path)
            popup.dismiss()
        
    def classify_image(self, path):
        self.image.source = path
        label, confidence = self.classifier.predict(path)
        self.label.text = f"Disease: {label} {confidence:.2f}"
        self.clear_widgets()
        self.add_widget(self.image)
        self.add_widget(self.label)
        self.add_widget(self.button)
        
class LeafAppMain(App):
    def build(self):
        return LeafApp()

if __name__ == "__main__":
    LeafAppMain().run()
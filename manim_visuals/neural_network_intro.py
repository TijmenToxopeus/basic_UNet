from manim import *


class NeuralNetworkIntro(Scene):
    def construct(self):
        input_neurons = 3
        hidden_neurons = 4
        output_neurons = 2

        layer_x_positions = [-4, 0, 4]
        neuron_radius = 0.25

        def create_layer(n_neurons, x_pos):
            layer = VGroup()
            for i in range(n_neurons):
                neuron = Circle(radius=neuron_radius)
                neuron.move_to([x_pos, (n_neurons - 1) / 2 - i, 0])
                layer.add(neuron)
            return layer

        def connect_layers(layer1, layer2):
            connections = VGroup()
            for n1 in layer1:
                for n2 in layer2:
                    connections.add(Line(n1.get_center(), n2.get_center()))
            return connections

        input_layer = create_layer(input_neurons, layer_x_positions[0])
        hidden_layer = create_layer(hidden_neurons, layer_x_positions[1])
        output_layer = create_layer(output_neurons, layer_x_positions[2])

        connections = VGroup(
            connect_layers(input_layer, hidden_layer),
            connect_layers(hidden_layer, output_layer)
        )

        self.play(Create(input_layer), Create(hidden_layer), Create(output_layer))
        self.play(Create(connections))
        self.wait()

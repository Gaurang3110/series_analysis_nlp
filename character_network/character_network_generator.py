import pandas as pd
from pyvis.network import Network
import networkx as nx


class CharacterNetworkGenerator:
    def __init__(self, window_size=10):
        self.window_size = window_size  # Configurable window size

    def generate_character_network(self, df):  # Added self parameter
        """
        Generate character co-occurrence network from DataFrame with NER results

        Args:
            df: DataFrame containing 'ners' column with lists of character sets

        Returns:
            DataFrame with source, target, and count of co-occurrences
        """
        entity_relationships = []

        for row in df['ners']:
            previous_entities_in_window = []

            for sentence in row:
                # Add current sentence's entities to window
                current_entities = list(sentence)
                previous_entities_in_window.append(current_entities)

                # Keep only the last N sentences
                previous_entities_in_window = previous_entities_in_window[-self.window_size:]

                # Flatten the window
                flattened_entities = sum(previous_entities_in_window, [])

                # Create relationships between current and previous entities
                for entity in current_entities:
                    for other_entity in flattened_entities:
                        if entity != other_entity:
                            relationship = tuple(sorted((entity, other_entity)))
                            entity_relationships.append(relationship)

        # Convert to DataFrame and count relationships
        relationship_df = pd.DataFrame(entity_relationships, columns=['source', 'target'])
        relationship_df = (relationship_df
                           .groupby(['source', 'target'])
                           .size()
                           .reset_index(name='count')
                           .sort_values('count', ascending=False))

        return relationship_df

    def draw_network_graph(self, relationship_df, top_n=200):
        """
        Visualize the character network using PyVis

        Args:
            relationship_df: DataFrame from generate_character_network()
            top_n: Number of top relationships to visualize

        Returns:
            HTML iframe containing the network visualization
        """
        # Filter top relationships
        relationship_df = relationship_df.head(top_n)

        # Create network graph
        G = nx.from_pandas_edgelist(
            relationship_df,
            source='source',
            target='target',
            edge_attr='count',
            create_using=nx.Graph()
        )

        # Configure visualization
        net = Network(
            notebook=True,
            width="1000px",
            height="700px",
            bgcolor="#222222",
            font_color="white",
            cdn_resources="remote"
        )

        # Set node sizes based on degree
        node_degree = dict(G.degree)
        nx.set_node_attributes(G, node_degree, 'size')

        # Generate HTML
        net.from_nx(G)
        net.toggle_physics(True)
        html = net.generate_html()
        html = html.replace("'", "\"")

        # Create responsive iframe
        iframe_html = f"""
        <iframe style="width: 100%; height: 600px; margin: 0 auto" 
                name="result" 
                allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" 
                sandbox="allow-modals allow-forms allow-scripts allow-same-origin 
                         allow-popups allow-top-navigation-by-user-activation allow-downloads" 
                allowfullscreen="" 
                allowpaymentrequest="" 
                frameborder="0" 
                srcdoc='{html}'>
        </iframe>
        """

        return iframe_html
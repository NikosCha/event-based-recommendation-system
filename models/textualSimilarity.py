class TextualModel:
    def __init__(self, graph):

        hub_url = '/tmp/module/universal_module/'
        self.embed = hub.KerasLayer(hub_url)
        self.graph = graph
        self.session = tf.compat.v1.Session(config=None, graph=graph)
    def cleanhtml(self, html): 
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()

            #clean html tags etc from descriptions
            randomUserTraining.loc[:,'description'] = randomUserTraining.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomUserTesting.loc[:,'description'] = randomUserTesting.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomEvents.loc[:,'description'] = randomEvents.apply(lambda row: self.cleanhtml(row.description), axis=1)


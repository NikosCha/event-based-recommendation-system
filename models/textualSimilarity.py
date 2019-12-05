    def cleanhtml(self, html): 
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()

            #clean html tags etc from descriptions
            randomUserTraining.loc[:,'description'] = randomUserTraining.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomUserTesting.loc[:,'description'] = randomUserTesting.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomEvents.loc[:,'description'] = randomEvents.apply(lambda row: self.cleanhtml(row.description), axis=1)


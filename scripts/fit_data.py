from scripts.pipeline import BoxOfficeFeatures


def main():
    record = BoxOfficeFeatures(
        title="Star Wars: The Force Awakens",
        release_year=2015,
        box_office=2.07e9,
        budget=3.06e8,
        rating="pg-13",
        collection="{'id': 10, 'name': 'Star Wars Collection', 'poster_path': '/ghd5zOQnDaDW1mxO7R5fXXpZMu.jpg', 'backdrop_path': '/d8duYyyC9J5T825Hg7grmaabfxQ.jpg'}",
        cast=["Harrison Ford", "Mark Hamill", "Carrie Fisher", "Adam Driver", "Daisy Ridley"],
        director="J.J. Abrams",
        writers=["Lawrence Kasdan", "J.J. Abrams", "Michael Arndt"],
        distributors=["Walt Disney Pictures", "Walt Disney Studios Motion Pictures"],
        genres=["Action", "Space opera", "Science Fiction", "Adventure", "Fantasy"],
        plot="""
            Thirty years after the Battle of Endor,[a] the First Order has risen from the fallen Galactic Empire and seeks to end the New Republic. The Resistance, led by General Leia Organa, opposes the First Order. Leia searches for her twin brother, Luke Skywalker, who is missing.
    
            On the desert planet Jakku, Resistance pilot Poe Dameron receives a map to Luke's location. First Order stormtroopers commanded by Kylo Ren arrive and capture Poe. His droid, BB-8, escapes with the map and encounters Rey, a lone scavenger. Kylo tortures Poe using the Force and learns of BB-8. Stormtrooper FN-2187, disillusioned with the First Order, saves Poe, and they escape in a stolen TIE fighter. Upon learning that FN-2187 has no other name, Poe names him "Finn". As they head to Jakku to retrieve BB-8, a First Order Star Destroyer shoots them, and they crash-land. Finn survives and assumes Poe was killed after finding his jacket in the wreck. Finn encounters Rey and BB-8, but the First Order tracks them and launches an airstrike. Rey, Finn, and BB-8 steal the Millennium Falcon and escape Jakku.
            
            The Falcon is discovered and boarded by Han Solo and Chewbacca. Gangs seeking to settle debts with Han attack, but the group escapes in the Falcon. At the First Order's Starkiller Base, a planet converted into a superweapon, Supreme Leader Snoke approves General Hux's request to use the weapon on the New Republic. Snoke questions Kylo's ability to deal with emotions surrounding his father, Han Solo, whom Kylo states means nothing to him.
            
            Aboard the Falcon, Han determines that BB-8's map is incomplete. He then explains that Luke attempted to rebuild the Jedi Order but exiled himself when an apprentice turned to the dark side, destroyed Luke's temple, and slaughtered the other apprentices. The crew travels to the planet Takodana and meets with cantina owner Maz Kanata, who offers help getting BB-8 to the Resistance. The Force draws Rey to a secluded vault, where she finds Anakin Skywalker's lightsaber. She experiences disturbing visions, including a childhood memory of a ship leaving her on Jakku. Rey denies the lightsaber at Maz's offering and flees into the woods. Maz gives Finn the lightsaber for safekeeping.
            
            Starkiller Base destroys the Hosnian star system, including the New Republic capital: Hosnian Prime, leaving the Resistance without support. The First Order attacks Takodana in search of BB-8. Han, Chewbacca, and Finn are saved by Resistance X-wing fighters led by Poe, who survived the crash. Leia arrives at Takodana with C-3PO and reunites with Han. Meanwhile, Kylo captures Rey, realizing she had seen the map, and takes her to Starkiller Base, but she resists his mind-reading attempts. Snoke orders Kylo to bring Rey to him. Discovering she can use the Force, Rey escapes using a Jedi mind trick on a stormtrooper guard.
            
            At the Resistance base, BB-8 finds R2-D2, who had been in low-power mode since Luke's disappearance. As Starkiller Base prepares to fire again, the Resistance plans to destroy it by attacking its thermal oscillator. Using the Falcon, Han, Chewbacca, and Finn infiltrate the facility, find Rey, and plant explosives. Han confronts Kylo, calling him by his birth name, Ben, and implores him to abandon the dark side. Kylo seems to consider this, but he ultimately kills Han. Chewbacca shoots Kylo, injuring him, and sets off the explosives, allowing Poe to attack and destroy the base's thermal oscillator.
            
            Kylo pursues Rey and Finn into the woods and incapacitates Rey. Finn uses the lightsaber to duel Kylo but is quickly defeated. Rey awakens, takes the lightsaber, and defeats Kylo in a duel. Snoke orders Hux to evacuate and bring Kylo to him to complete his training. Chewbacca saves Rey and the injured Finn, and they escape aboard the Falcon. As the Resistance forces flee, Starkiller Base implodes and erupts into a star. R2-D2 awakens and reveals the rest of the map, which points to the oceanic planet Ahch-To.
            
            Rey, Chewbacca, and R2-D2 travel to Ahch-To on the Falcon. Rey finds Luke atop a cliff on a remote island and presents him with his lightsaber.
        """
    )
    pass

if __name__ == "__main__":
    main()
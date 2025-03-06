import pytest
import sys, os
sys.path.append(os.path.abspath('/app'))
from app.mediatree_utils import *
import modin.pandas as pd
from datetime import datetime
import logging

df_with_misinformation = pd.DataFrame([{
            "plaintext": "quatorze pour cent quinze pour cent de voitures électriques vendues en europe l' année dernière à cette année treize pour cent dont vous voyez en plus la si vous voulez parce que c' est trop cher et parce que en france une fin en europe une fois n' est pas coutume on a un problème de souveraineté et à dire que tout ce qu' elle est tout ce que sont les batteries électriques les méthodes qui permettent de sarah je viens progressivement mais de quoi rémi waquet jean-françois liée à un moment donné il faudra quand même qu' au niveau mondial on se pose la vraie question mais c' est les accords de paris un sol et la lutte mondiale contre les la production de gaz à effet de serre ça ne ça va ne pas va aller pas s' aller arrangeant si on renonce aux impératifs climatiques c' est tout c' est aussi bête que ça non mais on ne sait pas ce petit petit traité européen mine de rien quand on voit ce qu' on pèse dans le monde euh prendre des mesures qui vient pénaliser une industrie aussi consommateurs nan mais sans aller dans le mur jean-françois mais quand on voit ce que fait la chine quand on voit ce que fait la quand on voit maintenant avec donald trump ce que fait ce que ce que font les etats unis sans dire que c' est le modèle absolu on ne peut pas se dire nous absolument on va essayer de faire encore plus vertueux que les plus vertueux on a une on a une a on aujourd' a hui on aujourd' on hui est on les est leaders les dans leaders la dans transition la énergétique transition on énergétique a une énergie propre que le nucléaire et on ne peut pas se laisser si vous voulez imposer des normes notamment sur ces domaines là par les allemands qui ont arrêté le nucléaire parce que les chardons c' est aussi une part de la réalité est tout le paradoxe de l' europe est unie pour pas mal de sujets et un mais vous avez des centrales à charbon en allemagne et des centrales nucléaires en france voisines hanks est tout le paradoxe d' une europe qui n' arrive pas à s' harmoniser joint pas assez vite pour la production ça vaudra aussi c' est le débat qui coup sur oui la bien défense sûr européenne bon sur après bon c' après c' est vrai que vous l' avez l' avez dit il y a ce problème de coût c' est à dire que euh les postures françaises sont trente pour cent plus chère pour les voitures européennes quoi bien sûr que je mets mes élèves et les chinois attention parce qu' ils ne sont pas allées dans le tout électrique non plus du tout euh ils ont toute la gamme et après ils vont vous car euh ce qui fonctionne ce qui ne fonctionne pas toyota le leader mondial lui n' est pas allé sur le tout électrique non plus il est allé sur l' hybride l' hybride qui revient dans la course mais évidemment bon hier les questions prioritaires avec des marques chinoises qui pratiquent des conditions",
            "start": "2025-03-05T07:20:00+00:00",
            "channel_title": "Sud Radio",
            "channel_name": "sud-radio",
            "channel_program": "Le Grand Matin",
            "channel_program_type": "Information - Magazine",
            "model_name": "my_model",
            "model_result": 10,
            "model_reason": "Le texte affirme que la France doit continuer à utiliser l'énergie nucléaire pour atteindre ses objectifs climatiques, tout en critiquant les normes imposées par d'autres pays européens. Cela constitue une promotion de l'idée que le nucléaire est la seule solution viable pour la transition énergétique, ce qui va à l'encontre des consensus scientifiques sur la nécessité d'une approche diversifiée pour lutter contre le changement climatique.",
            "year": 2025,
            "month": 3,
            "day": 5,
            "channel": "sud-radio",
            "url_mediatree": "https://keywords.mediatree.fr/player/?fifo=sud-radio&start_cts=1741159200&end_cts=1741159320&position_cts=1741159200"
    }])

df_with_misinformation_video = pd.DataFrame([{
            "plaintext": "faire entendre une voix différente et là on est au cur d' une des raisons pour lesquelles ça marche je pense bien c' est que les uns et les hôtes savent que ce qu' ils écoutent ici c' est toutes les voies bien sûr que quand il faut dire les choses dans la vie de vie de dix sur le covid a faut le dire sur la vaccination faut le dire c' est dessins tout dans mon livre c' était hyper compliqué par le journaliste de faire cette couverture de crise sanitaire parce que le moment vous toucher les sujets qui qui va dehors de générale ligne générale de de l' état vous êtes tout au-dessus de suite décrédibilisé diabolisé par les journalistes de plateaux de télé stream à des experts donc je pense que c' était compliqué pour tous mais c' est vrai qu' il a eu les chaînes qui sont vraiment pas faire un bon job avec la couverture de cette crise autre sujet sensible cette entend censure qui existe aussi je pense que c' est très important de parler je ne donne pas les leçons dans mon livre c' est pas un matelas je suis heureuse mais je le dont point le de point vue de différent vue j' différent j' espère que le quai ceux qui vont lire peut réfléchir un petit peu sur les sujets qui jetés reste quarante secondes je ne voudrais pas qu' on soit en retard parce qu' on est en retard tous les jours avec jean-marc morandini donc si vous avez dix secondes dix secondes et bien il y a du grand journalisme le wall street journal avait révélé l' intrus que le new-york times sort en quête aujourd' hui sur les douze bases de la cia qui depuis deux mille quatorze espionner la russie avec huit cents agents financer ça c' est une véritable enquête king journaliste me permet d' accéder à un peu de vérité je en vous français parle français je parle vous des parle journalistes des journalistes français et je parle des journalistes français on passe notre temps à donner des leçons de la russie on passe notre temps à donner des leçons mais sur la vaccination ne veut pas en parler euh parfois euh et on a expliqué aux gens de seconde on a on a indiqué aujourd' il fallait se vacciner parce que le vaccin empêcher la transmission et puis c' était pas vrai ce n' était pas vrai donc on a envoyé tous les gosses se faire vacciner avec parfois des effets secondaires mais voilà alors qu' on vaccine les anciens il y avait pas de souci mais il faut dire la vérité cette vérité on ne veut pas forcément",
            "start": "2025-03-05T09:34:00+00:00",
            "channel_title": "CNews",
            "channel_name": "itele",
            "channel_program": "Information en continu",
            "channel_program_type": "Information en continu",
            "model_name": "ft:gpt-4o-mini-2024-07-18:personal::B1xWiJRm",
            "model_result": 10,
            "model_reason": "L'affirmation selon laquelle le vaccin contre le COVID-19 ne prévenait pas la transmission est une désinformation qui contredit le consensus scientifique établi.",
            "year": 2025,
            "month": 3,
            "day": 5,
            "channel": "itele",
            "url_mediatree": "https://keywords.mediatree.fr/player/?fifo=itele&start_cts=1741167240&end_cts=1741167360&position_cts=1741167240"
            }])

def test_get_url_mediatree():
    date_string = "2024-12-12 10:10:10"
    date = datetime.fromisoformat(date_string)
    output = get_url_mediatree(channel="itele", date=date)
    assert output == "https://keywords.mediatree.fr/player/?fifo=itele&start_cts=1733998210&end_cts=1733998330&position_cts=1733998210"


def test_get_video_urls(monkeypatch):
    def mock_get(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {"src": "https://example.com/video.mp4"}
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)

    videos_urls = get_video_urls(df_with_misinformation)
    assert videos_urls[0] == "https://example.com/video.mp4"


def test_get_new_plaintext_from_whisper_mp3():
    result = get_new_plaintext_from_whisper(df_with_misinformation)
    assert False == True

def test_get_new_plaintext_from_whisper_mp4():
    result = get_new_plaintext_from_whisper(df_with_misinformation_video)
    assert False == True
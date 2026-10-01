#Tests for shared.signatures (Stage 1b)

#Fixtures are synthetic signature blocks and messages --> no real PII


from shared.signatures import signature_zones


def _spans(text, *pairs):
    #(substring, entity) pairs, each found after the previous one --> (start, end, entity) redactions
    out, at = [], 0
    for sub, entity in pairs:
        i = text.index(sub, at)
        out.append((i, i + len(sub), entity))
        at = i + len(sub)
    return out


def _stripped(text, *pairs):
    #what goes: the zone's text, or None when the body is left alone
    zones = signature_zones(text, _spans(text, *pairs))
    assert len(zones) <= 1
    return text[zones[0][0]:zones[0][1]] if zones else None


MDWP = ("Sounds great, see you then!\n"
        "*Happy Planning!!!*Miley Durian Heisman\n"
        "MDWP - Miley Durian Wedding Planner\n"
        "miley@example.com\n"
        "730-153-0100\n"
        "Official Blog and Website: example.com\n"
        "Please add and \"Like\" us on Facebook\n"
        "Check us out Yelp\n")
MDWP_SPANS = (("Miley", "PERSON"), ("Miley Durian", "PERSON"), ("miley@example.com", "EMAIL_ADDRESS"),
              ("730-153-0100", "PHONE_NUMBER"), ("example.com", "URL"))

HEALTH = ("Can you confirm the start time?\n"
         "Thanks,\n"
         "Patrick Star, MBA\n"
         "Senior Data Engineer\n"
         "HMM- Wood Level\n"
         "\n"
         "Health Labs | Action from Insight | 105 Main Street | Balenciaga CA 91341 | phone 661.555.0199 | "
         "patrick@example.com\n"
         "Health Labs is in-network with major plans, and continues as a national preferred lab for many clients.\n")
HEALTH_SPANS = (("Patrick Star", "PERSON"), ("Health Labs", "ORGANIZATION"), ("105 Main Street", "STREET_ADDRESS"),
               ("Balenciaga", "LOCATION"), ("91341", "LOCATION"), ("661.555.0199", "PHONE_NUMBER"),
               ("patrick@example.com", "EMAIL_ADDRESS"), ("Health Labs", "ORGANIZATION"))


# what goes

class TestStripped:

    def test_planner_signature_goes_whole(self):
        #titles, abbreviations and taglines aren't PII entities, so only the block as a unit catches them
        gone = _stripped(MDWP, *MDWP_SPANS)
        assert gone.startswith("*Happy Planning!!!*")
        assert all(t in gone for t in ("MDWP", "miley@example.com", "730-153-0100", "Official Blog", "Check us out"))

    def test_letterhead_with_a_corporate_tagline_goes(self):
        #the tagline reads like a sentence, but it's impersonal and sits below the contact details
        gone = _stripped(HEALTH, *HEALTH_SPANS)
        assert gone.startswith("Thanks,")
        assert all(t in gone for t in ("MBA", "Senior Data Engineer", "HMM- Wood Level", "Health Labs",
                                       "Action from Insight", "105 Main Street", "Balenciaga CA 91341", 
                                       "phone 661.555.0199", "is in-network"))

    def test_disclaimer_after_a_signature_goes_with_it(self):
        body = ("See you Saturday!\nBest,\nAnna Lee\n714-555-0101 | anna@example.com\n\n"
                "CONFIDENTIALITY NOTICE: This e-mail may contain confidential information intended only "
                "for the recipient named above.\n")
        gone = _stripped(body, ("Anna Lee", "PERSON"), ("714-555-0101", "PHONE_NUMBER"),
                         ("anna@example.com", "EMAIL_ADDRESS"))
        assert gone.startswith("Best,") and "CONFIDENTIALITY" in gone


# what stays

class TestKept:

    def test_question_next_to_a_phone_number(self):
        body = "Hi, can you perform on the 22nd? My number is 714-555-0102 if easier.\nMaria Tran\n"
        assert _stripped(body, ("714-555-0102", "PHONE_NUMBER"), ("Maria Tran", "PERSON")) is None

    def test_question_after_the_contact_block(self):
        #the sender keeps talking after it, so it isn't a sign-off
        body = "Anna Lee\n714-555-0101\nanna@example.com\nAlso, do you have availability in March?\n"
        assert _stripped(body, ("Anna Lee", "PERSON"), ("714-555-0101", "PHONE_NUMBER"),
                         ("anna@example.com", "EMAIL_ADDRESS")) is None

    def test_no_contact_details_no_zone(self):
        #a name and two cities are a message, not a signature
        body = "Hi Anna, we perform in Irvine and Tustin most weekends.\nThanks,\nTom\n"
        assert _stripped(body, ("Anna", "PERSON"), ("Irvine", "LOCATION"), ("Tustin", "LOCATION"),
                         ("Tom", "PERSON")) is None

    def test_event_fields_stay_and_the_signature_below_goes(self):
        body = ("Below are the details:\nWedding Date: Friday, November 27th\nVenue: Casa Rosa\n"
                "Coordinator Name: Jane Roe\nTime for Performance: 7:15 PM\nDate of wedding: Saturday\n"
                "Thanks,\nJane Roe\nRoe Events | 714-555-0103 | jane@example.com\n")
        gone = _stripped(body, ("Casa Rosa", "LOCATION"), ("Jane Roe", "PERSON"), ("Jane Roe", "PERSON"),
                         ("Roe Events", "ORGANIZATION"), ("714-555-0103", "PHONE_NUMBER"),
                         ("jane@example.com", "EMAIL_ADDRESS"))
        assert gone == "Thanks,\nJane Roe\nRoe Events | 714-555-0103 | jane@example.com\n"

    def test_website_form_is_left_alone(self):
        body = ("Name: Jane Roe\nEmail: jane@example.com\nPhone: 714-555-0104\nDate/Time of Event: March 3\n"
                "Location: Irvine\nMessage (Please include details): Two lions please\n")
        assert _stripped(body, ("Jane Roe", "PERSON"), ("jane@example.com", "EMAIL_ADDRESS"),
                         ("714-555-0104", "PHONE_NUMBER"), ("Irvine", "LOCATION")) is None

    def test_wrapped_line_keeps_the_protection_of_the_line_it_continues(self):
        #"covers the gas..." has no pronoun of its own; it's the second half of the sentence above
        body = ("Anna Lee | 714-555-0101\n"
                "We charge a travel fee of fifty dollars for events outside the county which only\n"
                "covers the gas for three cars from the studio.\n"
                "Anna Lee Events | anna@example.com\n")
        gone = _stripped(body, ("Anna Lee", "PERSON"), ("714-555-0101", "PHONE_NUMBER"),
                         ("Anna Lee Events", "ORGANIZATION"), ("anna@example.com", "EMAIL_ADDRESS"))
        assert gone == "Anna Lee Events | anna@example.com\n"

    def test_message_above_a_sign_off_stays(self):
        #"Please see attached..." is short and names no one, but it's above "Best,"
        body = ("Hi Anna,\nPlease see attached for the contract as well.\nBest,\nTom Nguyen\n"
                "Manager | Lion Dance Co\ntom@example.com | 714-555-0105\n")
        gone = _stripped(body, ("Anna", "PERSON"), ("Tom Nguyen", "PERSON"),
                         ("tom@example.com", "EMAIL_ADDRESS"), ("714-555-0105", "PHONE_NUMBER"))
        assert gone.startswith("Best,") and "Please see attached" not in gone

    def test_vietnamese_message_stays(self):
        body = ("Chào anh Minh,\nChúng tôi muốn đặt hai con lân cho lễ khai trương tuần sau nhé anh.\n"
                "Liên Nguyễn\n714-555-0106 | lien@example.com\n")
        gone = _stripped(body, ("Minh", "PERSON"), ("Liên Nguyễn", "PERSON"), ("714-555-0106", "PHONE_NUMBER"),
                         ("lien@example.com", "EMAIL_ADDRESS"))
        assert gone == "Liên Nguyễn\n714-555-0106 | lien@example.com\n"

    def test_label_line_above_is_not_pulled_in(self):
        #"My phone number is:" introduces the value below it
        body = "My phone number is:\n714-555-0107\nJo Tran | jo@example.com\n"
        gone = _stripped(body, ("714-555-0107", "PHONE_NUMBER"), ("Jo Tran", "PERSON"),
                         ("jo@example.com", "EMAIL_ADDRESS"))
        assert gone == "714-555-0107\nJo Tran | jo@example.com\n"

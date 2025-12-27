

def gen_history():
    facts = [
        {
            "q1": "Who was the first president of the United States?",
            "q2": "Name the first U.S. president.",
            "a": "George Washington was the first president of the United States.",
        },
        {
            "q1": "In what year did the United States declare independence?",
            "q2": "What year is associated with U.S. independence?",
            "a": "The United States declared independence in 1776.",
        },
        {
            "q1": "In what year did World War II end?",
            "q2": "What year marks the end of World War II?",
            "a": "World War II ended in 1945.",
        },
        {
            "q1": "In what year did World War I begin?",
            "q2": "What year marks the start of World War I?",
            "a": "World War I began in 1914.",
        },
        {
            "q1": "In what year did World War I end?",
            "q2": "What year marks the end of World War I?",
            "a": "World War I ended in 1918.",
        },
        {
            "q1": "In what year did the Berlin Wall fall?",
            "q2": "What year is associated with the fall of the Berlin Wall?",
            "a": "The Berlin Wall fell in 1989.",
        },
        {
            "q1": "In what year did humans first land on the Moon?",
            "q2": "What year was the first Moon landing?",
            "a": "Humans first landed on the Moon in 1969.",
        },
        {
            "q1": "Who developed the theory of relativity?",
            "q2": "The theory of relativity is associated with which scientist?",
            "a": "Albert Einstein developed the theory of relativity.",
        },
        {
            "q1": "Who was known as the Maid of Orleans?",
            "q2": "The nickname 'Maid of Orleans' refers to whom?",
            "a": "Joan of Arc was known as the Maid of Orleans.",
        },
        {
            "q1": "Who wrote the Declaration of Independence?",
            "q2": "Who was the principal author of the U.S. Declaration of Independence?",
            "a": "Thomas Jefferson was the principal author of the Declaration of Independence.",
        },
        {
            "q1": "In what year was the United Nations founded?",
            "q2": "What year marks the founding of the United Nations?",
            "a": "The United Nations was founded in 1945.",
        },
        {
            "q1": "What document limited the power of the English king in 1215?",
            "q2": "Which charter signed in 1215 limited the English king's power?",
            "a": "The Magna Carta limited the power of the English king in 1215.",
        },
        {
            "q1": "Where did the Renaissance begin?",
            "q2": "The Renaissance started in which region?",
            "a": "The Renaissance began in Italy.",
        },
        {
            "q1": "Which civilization built the pyramids at Giza?",
            "q2": "The pyramids at Giza were built by which civilization?",
            "a": "The pyramids at Giza were built by ancient Egyptians.",
        },
        {
            "q1": "Which civilization is associated with the city of Rome?",
            "q2": "The city of Rome was the center of which civilization?",
            "a": "Rome was the center of the Roman civilization.",
        },
        {
            "q1": "Who was the first man in space?",
            "q2": "Name the first human to travel into space.",
            "a": "Yuri Gagarin was the first man in space.",
        },
        {
            "q1": "In what year did the American Civil War begin?",
            "q2": "What year marks the start of the American Civil War?",
            "a": "The American Civil War began in 1861.",
        },
        {
            "q1": "In what year did the American Civil War end?",
            "q2": "What year marks the end of the American Civil War?",
            "a": "The American Civil War ended in 1865.",
        },
        {
            "q1": "Who invented the printing press in Europe?",
            "q2": "Which inventor is associated with the printing press in Europe?",
            "a": "Johannes Gutenberg is associated with the printing press in Europe.",
        },
        {
            "q1": "What was the ship that sank in 1912 after hitting an iceberg?",
            "q2": "Which famous ship sank in 1912 on its maiden voyage?",
            "a": "The Titanic sank in 1912 after hitting an iceberg.",
        },
        {
            "q1": "What empire was ruled by Genghis Khan?",
            "q2": "Genghis Khan led which empire?",
            "a": "Genghis Khan led the Mongol Empire.",
        },
        {
            "q1": "Who was the leader of the Indian independence movement known for nonviolence?",
            "q2": "Which leader is famous for nonviolent resistance in India?",
            "a": "Mahatma Gandhi led India's independence movement with nonviolent resistance.",
        },
        {
            "q1": "What was the name of the period of rapid industrial growth starting in the 18th century?",
            "q2": "Which period is known for major industrial and technological change starting in the 1700s?",
            "a": "It is called the Industrial Revolution.",
        },
        {
            "q1": "Which ancient Greek city-state is known for democracy?",
            "q2": "Democracy in ancient Greece is most associated with which city-state?",
            "a": "Athens is most associated with democracy in ancient Greece.",
        },
        {
            "q1": "Who delivered the 'I Have a Dream' speech?",
            "q2": "The 'I Have a Dream' speech was given by whom?",
            "a": "Martin Luther King Jr. delivered the 'I Have a Dream' speech.",
        },

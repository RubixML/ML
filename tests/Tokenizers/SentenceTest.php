<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Sentence;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Sentence
 */
class SentenceTest extends TestCase
{
    /**
     * @var Sentence
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new Sentence();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Sentence::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     * @dataProvider tokenizeProvider
     *
     * @param string $text
     * @param list<string> $expected
     */
    public function tokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function tokenizeProvider() : Generator
    {
        /**
         * English
         */
        yield [
            "Here's to the crazy ones. The misfits. The rebels. The troublemakers. The round pegs in the square holes. The ones who see things differently. They're not fond of rules. And they have no respect for the status quo. You can quote them, disagree with them, glorify or vilify them. About the only thing you can't do is ignore them. Because they change things. They push the human race forward. And while some may see them as the crazy ones, we see genius. Because the people who are crazy enough to think they can change the world, are the ones who do.",
            [
                "Here's to the crazy ones.",
                'The misfits.',
                'The rebels.',
                'The troublemakers.',
                'The round pegs in the square holes.',
                'The ones who see things differently.',
                "They're not fond of rules.",
                'And they have no respect for the status quo.',
                'You can quote them, disagree with them, glorify or vilify them.',
                "About the only thing you can't do is ignore them.",
                'Because they change things.',
                'They push the human race forward.',
                'And while some may see them as the crazy ones, we see genius.',
                'Because the people who are crazy enough to think they can change the world, are the ones who do.',
            ],
        ];

        /**
         * Spanish
         */
        yield [
            'Aquí están los locos. Los inadaptados. Los rebeldes. Los alborotadores. Las clavijas redondas en los agujeros cuadrados. Los que ven las cosas de manera diferente. No les gustan las reglas. Y no tienen respeto por el status quo. Puedes citarlos, estar en desacuerdo con ellos, glorificarlos o vilipendiarlos. Lo único que no puedes hacer es ignorarlos. Porque cambian las cosas. Empujan a la raza humana hacia adelante. Y mientras que algunos pueden verlos como los locos, nosotros vemos genio. Porque las personas que están lo suficientemente locas como para pensar que pueden cambiar el mundo, son las que lo hacen.',
            [
                'Aquí están los locos.',
                'Los inadaptados.',
                'Los rebeldes.',
                'Los alborotadores.',
                'Las clavijas redondas en los agujeros cuadrados.',
                'Los que ven las cosas de manera diferente.',
                'No les gustan las reglas.',
                'Y no tienen respeto por el status quo.',
                'Puedes citarlos, estar en desacuerdo con ellos, glorificarlos o vilipendiarlos.',
                'Lo único que no puedes hacer es ignorarlos.',
                'Porque cambian las cosas.',
                'Empujan a la raza humana hacia adelante.',
                'Y mientras que algunos pueden verlos como los locos, nosotros vemos genio.',
                'Porque las personas que están lo suficientemente locas como para pensar que pueden cambiar el mundo, son las que lo hacen.'
            ],
        ];

        /**
         * German
         */
        yield [
            'Ein Hoch auf die Verrückten. Die Außenseiter. Die Rebellen. Die Unruhestifter. Die runden Stifte in den quadratischen Löchern. Diejenigen, die die Dinge anders sehen. Sie mögen keine Regeln. Und sie haben keinen Respekt vor dem Status quo. Man kann sie zitieren, ihnen widersprechen, sie verherrlichen oder verunglimpfen. Das Einzige, was Sie nicht tun können, ist, sie zu ignorieren. Weil sie Dinge verändern. Sie treiben die Menschheit voran. Und während einige sie für die Verrückten halten, sehen wir Genies. Denn die Menschen, die verrückt genug sind zu glauben, dass sie die Welt verändern können, sind diejenigen, die es tun.',
            [
                'Ein Hoch auf die Verrückten.',
                'Die Außenseiter.',
                'Die Rebellen.',
                'Die Unruhestifter.',
                'Die runden Stifte in den quadratischen Löchern.',
                'Diejenigen, die die Dinge anders sehen.',
                'Sie mögen keine Regeln.',
                'Und sie haben keinen Respekt vor dem Status quo.',
                'Man kann sie zitieren, ihnen widersprechen, sie verherrlichen oder verunglimpfen.',
                'Das Einzige, was Sie nicht tun können, ist, sie zu ignorieren.',
                'Weil sie Dinge verändern.',
                'Sie treiben die Menschheit voran.',
                'Und während einige sie für die Verrückten halten, sehen wir Genies.',
                'Denn die Menschen, die verrückt genug sind zu glauben, dass sie die Welt verändern können, sind diejenigen, die es tun.',
            ],
        ];

        /**
         * French
         */
        yield [
            'Voici les fous. Les inadaptés. Les rebelles. Les fauteurs de troubles. Les chevilles rondes dans les trous carrés. Ceux qui voient les choses différemment. Ils n’aiment pas les règles. Et ils n’ont aucun respect pour le statu quo. Vous pouvez les citer, ne pas être d’accord avec eux, les glorifier ou les vilipender. La seule chose que vous ne pouvez pas faire est de les ignorer. Parce qu’ils changent les choses. Ils poussent la race humaine vers l’avant. Et tandis que certains peuvent les voir comme les fous, nous voyons du génie. Parce que les gens qui sont assez fous pour penser qu’ils peuvent changer le monde, sont ceux qui le font.',
            [
                'Voici les fous.',
                'Les inadaptés.',
                'Les rebelles.',
                'Les fauteurs de troubles.',
                'Les chevilles rondes dans les trous carrés.',
                'Ceux qui voient les choses différemment.',
                'Ils n’aiment pas les règles.',
                'Et ils n’ont aucun respect pour le statu quo.',
                'Vous pouvez les citer, ne pas être d’accord avec eux, les glorifier ou les vilipender.',
                'La seule chose que vous ne pouvez pas faire est de les ignorer.',
                'Parce qu’ils changent les choses.',
                'Ils poussent la race humaine vers l’avant.',
                'Et tandis que certains peuvent les voir comme les fous, nous voyons du génie.',
                'Parce que les gens qui sont assez fous pour penser qu’ils peuvent changer le monde, sont ceux qui le font.',
            ],
        ];

        /**
         * Russian
         */
        yield [
            'Вот к сумасшедшим. Неудачники. Повстанцы. Возмутители спокойствия. Круглые колышки в квадратных отверстиях. Те, кто видит вещи по-другому. Они не любят правила. И у них нет никакого уважения к статус-кво. Вы можете цитировать их, не соглашаться с ними, прославлять или поносить их. Единственное, что вы не можете сделать, это игнорировать их. Потому что они что-то меняют. Они толкают человеческую расу вперед. И хотя некоторые могут считать их сумасшедшими, мы видим гениев. Потому что люди, которые достаточно сумасшедшие, чтобы думать, что они могут изменить мир, - это те, кто это делает.',
            [
                'Вот к сумасшедшим.',
                'Неудачники.',
                'Повстанцы.',
                'Возмутители спокойствия.',
                'Круглые колышки в квадратных отверстиях.',
                'Те, кто видит вещи по-другому.',
                'Они не любят правила.',
                'И у них нет никакого уважения к статус-кво.',
                'Вы можете цитировать их, не соглашаться с ними, прославлять или поносить их.',
                'Единственное, что вы не можете сделать, это игнорировать их.',
                'Потому что они что-то меняют.',
                'Они толкают человеческую расу вперед.',
                'И хотя некоторые могут считать их сумасшедшими, мы видим гениев.',
                'Потому что люди, которые достаточно сумасшедшие, чтобы думать, что они могут изменить мир, - это те, кто это делает.',
            ],
        ];

        /**
         * Japanese
         */
        yield [
            'これがクレイジーなものです。不適合。反乱軍。トラブルメーカー。四角い穴の丸いペグ。物事を違った見方をする人。彼らはルールが好きではありません。そして、彼らは現状を尊重していません。あなたはそれらを引用したり、それらに同意しなかったり、それらを美化または非難したりすることができます。あなたができない唯一のことはそれらを無視することです。彼らは物事を変えるからです。彼らは人類を前進させます。そして、彼らをクレイジーなものと見なす人もいるかもしれませんが、私たちは天才を見ています。なぜなら、世界を変えることができると考えるほどクレイジーな人々は、そうする人々だからです。',
            [
                'これがクレイジーなものです。不適合。反乱軍。トラブルメーカー。四角い穴の丸いペグ。物事を違った見方をする人。彼らはルールが好きではありません。そして、彼らは現状を尊重していません。あなたはそれらを引用したり、それらに同意しなかったり、それらを美化または非難したりすることができます。あなたができない唯一のことはそれらを無視することです。彼らは物事を変えるからです。彼らは人類を前進させます。そして、彼らをクレイジーなものと見なす人もいるかもしれませんが、私たちは天才を見ています。なぜなら、世界を変えることができると考えるほどクレイジーな人々は、そうする人々だからです。',
            ],
        ];

        /**
         * Hindi
         */
        yield [
            'यहाँ पागल लोगों के लिए है. मिसफिट। विद्रोही। परेशानी पैदा करने वाले। चौकोर छेद में गोल खूंटे। जो चीजों को अलग तरह से देखते हैं। वे नियमों के शौकीन नहीं हैं। और उनके मन में यथास्थिति के लिए कोई सम्मान नहीं है। आप उन्हें उद्धृत कर सकते हैं, उनसे असहमत हो सकते हैं, उनका महिमामंडन कर सकते हैं या उन्हें बदनाम कर सकते हैं। केवल एक चीज जो आप नहीं कर सकते हैं वह है उन्हें अनदेखा करना। क्योंकि वे चीजों को बदलते हैं। वे मानव जाति को आगे बढ़ाते हैं। और जबकि कुछ उन्हें पागल लोगों के रूप में देख सकते हैं, हम प्रतिभा देखते हैं। क्योंकि जो लोग यह सोचने के लिए पर्याप्त पागल हैं कि वे दुनिया को बदल सकते हैं, वे ही हैं जो करते हैं।',
            [
                'यहाँ पागल लोगों के लिए है.',
                'मिसफिट। विद्रोही। परेशानी पैदा करने वाले। चौकोर छेद में गोल खूंटे। जो चीजों को अलग तरह से देखते हैं। वे नियमों के शौकीन नहीं हैं। और उनके मन में यथास्थिति के लिए कोई सम्मान नहीं है। आप उन्हें उद्धृत कर सकते हैं, उनसे असहमत हो सकते हैं, उनका महिमामंडन कर सकते हैं या उन्हें बदनाम कर सकते हैं। केवल एक चीज जो आप नहीं कर सकते हैं वह है उन्हें अनदेखा करना। क्योंकि वे चीजों को बदलते हैं। वे मानव जाति को आगे बढ़ाते हैं। और जबकि कुछ उन्हें पागल लोगों के रूप में देख सकते हैं, हम प्रतिभा देखते हैं। क्योंकि जो लोग यह सोचने के लिए पर्याप्त पागल हैं कि वे दुनिया को बदल सकते हैं, वे ही हैं जो करते हैं।',
            ],
        ];

        /**
         * Farsi
         */
        yield [
            "من امروز ملاقات با دوستانم را لغو کردم، چراکه خیلی خسته هستم! بعد از چند ماه مجدداً به دیدار خانواده‌ام می‌روم. آیا این برای من خوب خواهد بود؟آیا توانستی به من کمک کنی؟\nاین کتاب بسیار جالب است! \"با توجه به شرایطی که الان داریم، آیا می‌توانیم به یک قرار ملاقات برسیم\"؟",
            [
                'من امروز ملاقات با دوستانم را لغو کردم، چراکه خیلی خسته هستم!',
                'بعد از چند ماه مجدداً به دیدار خانواده‌ام می‌روم.',
                'آیا این برای من خوب خواهد بود؟',
                'آیا توانستی به من کمک کنی؟',
                'این کتاب بسیار جالب است!',
                '"با توجه به شرایطی که الان داریم، آیا می‌توانیم به یک قرار ملاقات برسیم"؟'
            ],
        ];

        /**
         * Chinese
         */
        yield [
            '这是给疯狂的人。格格不入的人。叛军。麻烦制造者。方孔中的圆钉。那些看待事物不同的人。他们不喜欢规则。他们不尊重现状。你可以引用它们，不同意它们，美化或诋毁它们。关于你唯一不能做的就是忽略它们。因为他们改变了事情。他们推动人类前进。虽然有些人可能认为他们是疯狂的，但我们看到的是天才。因为那些疯狂到认为自己可以改变世界的人，才是那些这样做的人。',
            [
                '这是给疯狂的人。格格不入的人。叛军。麻烦制造者。方孔中的圆钉。那些看待事物不同的人。他们不喜欢规则。他们不尊重现状。你可以引用它们，不同意它们，美化或诋毁它们。关于你唯一不能做的就是忽略它们。因为他们改变了事情。他们推动人类前进。虽然有些人可能认为他们是疯狂的，但我们看到的是天才。因为那些疯狂到认为自己可以改变世界的人，才是那些这样做的人。',
            ],
        ];

        /**
         * Arabic
         */
        yield [
            'هنا للمجانين. غير الأسوياء. المتمردون. مثيري الشغب. الأوتاد المستديرة في الثقوب المربعة. أولئك الذين يرون الأشياء بشكل مختلف. إنهم ليسوا مغرمين بالقواعد. وهم لا يحترمون الوضع الراهن. يمكنك اقتباسها أو الاختلاف معها أو تمجيدها أو تشويه سمعتها. حول الشيء الوحيد الذي لا يمكنك فعله هو تجاهلهم. لأنهم يغيرون الأشياء. إنهم يدفعون الجنس البشري إلى الأمام. وبينما قد يراهم البعض على أنهم مجانين ، فإننا نرى العبقرية. لأن الأشخاص المجانين بما يكفي للاعتقاد بأنهم يستطيعون تغيير العالم ، هم الذين يفعلون ذلك.',
            [
                'هنا للمجانين.',
                'غير الأسوياء.',
                'المتمردون.',
                'مثيري الشغب.',
                'الأوتاد المستديرة في الثقوب المربعة.',
                'أولئك الذين يرون الأشياء بشكل مختلف.',
                'إنهم ليسوا مغرمين بالقواعد.',
                'وهم لا يحترمون الوضع الراهن.',
                'يمكنك اقتباسها أو الاختلاف معها أو تمجيدها أو تشويه سمعتها.',
                'حول الشيء الوحيد الذي لا يمكنك فعله هو تجاهلهم.',
                'لأنهم يغيرون الأشياء.',
                'إنهم يدفعون الجنس البشري إلى الأمام.',
                'وبينما قد يراهم البعض على أنهم مجانين ، فإننا نرى العبقرية.',
                'لأن الأشخاص المجانين بما يكفي للاعتقاد بأنهم يستطيعون تغيير العالم ، هم الذين يفعلون ذلك.',
            ],
        ];
    }
}

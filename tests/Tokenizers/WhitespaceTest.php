<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\Whitespace;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(Whitespace::class)]
class WhitespaceTest extends TestCase
{
    protected Whitespace $tokenizer;

    public static function tokenizeProvider() : Generator
    {
        /**
         * English
         */
        yield [
            "If something's important enough, you should try. Even if - the probable outcome is failure.",
            [
                'If', "something's", 'important', 'enough,', 'you', 'should', 'try.',
                'Even', 'if', '-', 'the', 'probable', 'outcome', 'is', 'failure.',
            ],
        ];

        /**
         * Spanish
         */
        yield [
            'Si algo es lo suficientemente importante, deberías intentarlo. Incluso si - el resultado probable es el fracaso.',
            [
                'Si', 'algo', 'es', 'lo', 'suficientemente', 'importante,', 'deberías', 'intentarlo.',
                'Incluso', 'si', '-', 'el', 'resultado', 'probable', 'es', 'el', 'fracaso.',
            ],
        ];

        /**
         * German
         */
        yield [
            'Wenn etwas wichtig genug ist, sollten Sie es versuchen. Selbst wenn - das wahrscheinliche Ergebnis ist ein Scheitern.',
            [
                'Wenn', 'etwas', 'wichtig', 'genug', 'ist,', 'sollten', 'Sie', 'es', 'versuchen.',
                'Selbst', 'wenn', '-', 'das', 'wahrscheinliche', 'Ergebnis', 'ist', 'ein', 'Scheitern.',
            ],
        ];

        /**
         * French
         */
        yield [
            'Si quelque chose est assez important, vous devriez essayer. Même si - le résultat probable est l’échec.',
            [
                'Si', 'quelque', 'chose', 'est', 'assez', 'important,', 'vous', 'devriez', 'essayer.',
                'Même', 'si', '-', 'le', 'résultat', 'probable', 'est', 'l’échec.',
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->tokenizer = new Whitespace();
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('tokenizeProvider')]
    public function testTokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }
}

<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Tokenizers\Whitespace;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Whitespace
 */
class WhitespaceTest extends TestCase
{
    /**
     * @var Whitespace
     */
    protected $tokenizer;

    protected function setUp() : void
    {
        $this->tokenizer = new Whitespace();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Whitespace::class, $this->tokenizer);
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
}

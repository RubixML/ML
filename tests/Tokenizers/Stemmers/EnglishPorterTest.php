<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Tokenizers\Stemmers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\Stemmers\PorterEnglish;
use Rubix\ML\Tokenizers\Stemmers\Stemmer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(PorterEnglish::class)]
class PorterEnglishTest extends TestCase
{
    protected PorterEnglish $stemmer;

    /**
     * @return Generator<mixed[]>
     */
    public static function stemProvider() : Generator
    {
        yield ['caresses', 'caress'];
        yield ['ponies', 'poni'];
        yield ['ties', 'ti'];
        yield ['cats', 'cat'];
        yield ['feed', 'feed'];
        yield ['agreed', 'agre'];
        yield ['matting', 'mat'];
        yield ['mating', 'mate'];
        yield ['meeting', 'meet'];
        yield ['milling', 'mill'];
        yield ['messing', 'mess'];
        yield ['meetings', 'meet'];
        yield ['hopping', 'hop'];
        yield ['slaves', 'slave'];
        yield ['sliced', 'slice'];
        yield ['bidding', 'bid'];
        yield ['trotting', 'trot'];
        yield ['mopping', 'mop'];
        yield ['proceeding', 'proceed'];
        yield ['differing', 'differ'];
        yield ['sailing', 'sail'];
        yield ['sliding', 'slide'];
        yield ['agreeing', 'agre'];
        yield ['causing', 'caus'];
        yield ['speeding', 'speed'];
        yield ['careful', 'care'];
        yield ['carefully', 'carefulli'];
        yield ['conflated', 'conflat'];
        yield ['conflations', 'conflat'];
        yield ['troubled', 'troubl'];
        yield ['farming', 'farm'];
        yield ['caring', 'care'];
        yield ['caused', 'caus'];
        yield ['generous', 'gener'];
        yield ['general', 'gener'];
        yield ['organ', 'organ'];
        yield ['universe', 'univers'];
        yield ['herring', 'her'];
        yield ['skies', 'ski'];
        yield ['sky', 'sky'];
    }

    protected function setUp() : void
    {
        $this->stemmer = new PorterEnglish();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(PorterEnglish::class, $this->stemmer);
        $this->assertInstanceOf(Stemmer::class, $this->stemmer);
    }

    /**
     * @param string $word
     * @param string $expected
     */
    #[DataProvider('stemProvider')]
    #[Test]
    public function stem(string $word, string $expected) : void
    {
        $this->assertSame($expected, $this->stemmer->stem($word));
    }

    #[Test]
    public function toStringReturnsStemmer() : void
    {
        $this->assertSame('Porter English', $this->stemmer->__toString());
    }
}

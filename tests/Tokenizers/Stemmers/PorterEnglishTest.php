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
use ReflectionClass;

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
        yield ['relational', 'relat'];
        yield ['communicate', 'commun'];
        yield ['electricity', 'electr'];
        yield ['sensitivity', 'sensit'];
        yield ['triplicate', 'triplic'];
        yield ['formative', 'form'];
        yield ['electric', 'electr'];
        yield ['hopeful', 'hope'];
        yield ['goodness', 'good'];
        yield ['relieve', 'reliev'];
        yield ['cease', 'ceas'];
        yield ['controll', 'control'];
        yield ['adjustment', 'adjust'];
        yield ['operator', 'oper'];
        yield ['nationalization', 'nation'];
        yield ['revival', 'reviv'];
        yield ['inference', 'infer'];
        yield ['allowance', 'allow'];
        yield ['airliner', 'airlin'];
        yield ['probate', 'probat'];
        yield ['rate', 'rate'];
        yield ['died', 'di'];
        yield ['plastered', 'plaster'];
        yield ['bled', 'bled'];
        yield ['motoring', 'motor'];
        yield ['sing', 'sing'];
        yield ['sized', 'size'];
        yield ['tanned', 'tan'];
        yield ['falling', 'fall'];
        yield ['hissing', 'hiss'];
        yield ['fizzed', 'fizz'];
        yield ['failing', 'fail'];
        yield ['filing', 'file'];
        yield ['happy', 'happi'];
        yield ['valenci', 'valenc'];
        yield ['hesitanci', 'hesit'];
        yield ['digitizer', 'digit'];
        yield ['conformabli', 'conform'];
        yield ['radicalli', 'radic'];
        yield ['differentli', 'differ'];
        yield ['vileli', 'vile'];
        yield ['analogousli', 'analog'];
        yield ['vietnamization', 'vietnam'];
        yield ['predication', 'predic'];
        yield ['feudalism', 'feudal'];
        yield ['decisiveness', 'decis'];
        yield ['hopefulness', 'hope'];
        yield ['callousness', 'callous'];
        yield ['formaliti', 'formal'];
        yield ['sensitiviti', 'sensit'];
        yield ['sensibiliti', 'sensibl'];
        yield ['formalize', 'formal'];
        yield ['electriciti', 'electr'];
        yield ['electrical', 'electr'];
        yield ['adjustable', 'adjust'];
        yield ['defensible', 'defens'];
        yield ['irritant', 'irrit'];
        yield ['replacement', 'replac'];
        yield ['dependent', 'depend'];
        yield ['adoption', 'adopt'];
        yield ['homologou', 'homolog'];
        yield ['communism', 'commun'];
        yield ['activate', 'activ'];
        yield ['angulariti', 'angular'];
        yield ['homologous', 'homolog'];
        yield ['effective', 'effect'];
        yield ['bowdlerize', 'bowdler'];
        yield ['roll', 'roll'];
        yield ['inning', 'in'];
        yield ['outing', 'out'];
        yield ['canning', 'can'];
        yield ['earring', 'ear'];
        yield ['proceed', 'proce'];
        yield ['exceed', 'exce'];
        yield ['succeed', 'succe'];
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
    public function stemmerIsStateless() : void
    {
        $reflection = new ReflectionClass(PorterEnglish::class);

        $this->assertSame([], $reflection->getProperties());
    }

    #[Test]
    public function stemDoesNotRetainStateBetweenCalls() : void
    {
        foreach (self::stemProvider() as [$word, $expected]) {
            $this->assertSame($expected, $this->stemmer->stem($word));
            $this->assertSame($expected, (new PorterEnglish())->stem($word));
        }
    }

    #[Test]
    public function toStringReturnsStemmer() : void
    {
        $this->assertSame('Porter English', $this->stemmer->__toString());
    }
}

<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\TokenHashingVectorizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TokenHashingVectorizer::class)]
class TokenHashingVectorizerTest extends TestCase
{
    protected TokenHashingVectorizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new TokenHashingVectorizer(
            dimensions: 20,
            tokenizer: new Word(),
            hashFn: 'crc32'
        );
    }

    public function testTransform() : void
    {
        $dataset = Unlabeled::quick(samples: [
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [0, 1, 1, 0, 1, 1, 0, 4, 0, 1, 2, 1, 0, 0, 1, 1, 3, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}

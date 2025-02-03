<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\WordCountVectorizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(WordCountVectorizer::class)]
class WordCountVectorizerTest extends TestCase
{
    protected WordCountVectorizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new WordCountVectorizer(
            maxVocabularySize: 50,
            minDocumentCount: 1,
            maxDocumentRatio: 1.0,
            tokenizer: new Word()
        );
    }

    public function testFitTransform() : void
    {
        $dataset = Unlabeled::quick(samples: [
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $vocabulary = current($this->transformer->vocabularies() ?? []);

        $this->assertIsArray($vocabulary);
        $this->assertCount(20, $vocabulary);
        $this->assertContainsOnlyString($vocabulary);

        $dataset->apply($this->transformer);

        $expected = [
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}

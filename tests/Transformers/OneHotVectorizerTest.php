<?php

use Rubix\Engine\Transformers\OneHotVectorizer;
use Rubix\Engine\Tokenizers\WhitespaceTokenizer;
use Rubix\Engine\Math\Matrix;
use PHPUnit\Framework\TestCase;

class OneHotVectorizerTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $samples = ['the quick brown foxed jumped over the lazy man sitting at a bus stop drinking a can of coke'];

        $this->transformer = new OneHotVectorizer($samples);
    }

    public function test_vectorize_string()
    {
        $vector = $this->transformer->transform(['stop drinking coke stop'])[0];

        $this->assertTrue($vector instanceof Matrix);
        $this->assertEquals([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [0], [0], [1]], $vector->values());
    }
}

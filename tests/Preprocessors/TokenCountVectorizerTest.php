<?php

use Rubix\Engine\Preprocessors\TokenCountVectorizer;
use PHPUnit\Framework\TestCase;

class TokenCountVectorizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $data = [
            ['the quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of coke'],
        ];

        $this->preprocessor = new TokenCountVectorizer();

        $this->preprocessor->fit($data);
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(TokenCountVectorizer::class, $this->preprocessor);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['a quick bus jumped the lazy fox'],
            ['where are my friends'],
        ];

        $this->preprocessor->transform($data);

        $this->assertEquals([
            [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], $data);
    }

    public function test_vectorize_string()
    {
        $vector = $this->preprocessor->vectorize('stop drinking coke stop');

        $this->assertEquals([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1], $vector);
    }
}

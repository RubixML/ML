<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\JSON
 */
class JSONTest extends TestCase
{
    /**
     * @test
     */
    public function decode() : void
    {
        $data = (string) file_get_contents('tests/test.json');

        $actual = JSON::decode($data);

        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5, 'not monster'],
        ];

        $this->assertSame($expected, $actual);
    }

    /**
     * @test
     */
    public function encode() : void
    {
        $actual = JSON::encode(['package' => 'rubix/ml']);

        $expected = '{"package":"rubix\/ml"}';

        $this->assertSame($expected, $actual);
    }

    /**
     * @test
     */
    public function decodeBadData() : void
    {
        $this->expectException(RuntimeException::class);

        JSON::decode('[{"package":...}]');
    }
}

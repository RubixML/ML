<?php

namespace Rubix\ML\Tests\Helpers;

use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Helpers\JSON
 */
class JSONTest extends TestCase
{
    /**
     * @test
     */
    public function decode() : void
    {
        $actual = JSON::decode('{"attitude":"nice","texture":"furry","sociability":"friendly","rating":4,"class":"not monster"}');

        $expected = [
            'attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster',
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

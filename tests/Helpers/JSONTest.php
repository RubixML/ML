<?php

namespace Rubix\ML\Tests\Helpers;

use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\JSONException;
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
    public function encodeInvalidUTF8() : void
    {
        $this->expectException(JSONException::class);
        $this->expectExceptionMessage('Malformed UTF-8 characters, check encoding.');

        JSON::encode(['class' => "caf\xE9"]);
    }

    /**
     * @test
     */
    public function decodeNonArrayJson() : void
    {
        $this->expectException(JSONException::class);

        JSON::decode('42');
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

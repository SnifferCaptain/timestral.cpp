/*
 * @file preciseMath.glsl
 * @brief 双FP32余项补偿的exp与除法，不依赖FP64或subgroup宽度
 */

vec2 addPair(vec2 a, vec2 b) {
    precise float sum = a.x + b.x;
    precise float delta = sum - a.x;
    precise float error = (a.x - (sum - delta)) + (b.x - delta);
    error = error + a.y + b.y;
    precise float high = sum + error;
    precise float low = error - (high - sum);
    return vec2(high, low);
}

vec2 multiplyPair(vec2 a, float b) {
    precise float product = a.x * b;
    precise float error = fma(a.x, b, -product) + a.y * b;
    precise float high = product + error;
    precise float low = error - (high - product);
    return vec2(high, low);
}

float divideOrdered(float a, float b) {
    precise float quotient = a / b;
    if (isinf(quotient) || isnan(quotient) || isinf(b)) {
        return quotient;
    }
    precise float remainder = fma(-quotient, b, a);
    precise float correction = remainder / b;
    return quotient + correction;
}

float sqrtOrdered(float value) {
    precise float root = sqrt(value);
    if (root == 0.0 || isinf(root)) {
        return root;
    }
    precise float remainder = fma(-root, root, value);
    precise float correction = remainder / (2.0 * root);
    return root + correction;
}

float expOrdered(float value) {
    if (isnan(value)) {
        return value;
    }
    if (value > 88.72283905206835) {
        return uintBitsToFloat(0x7f800000u);
    }
    if (value < -103.97208404541016) {
        return 0.0;
    }
    precise float exponent = round(value * 1.4426950408889634);
    precise float high = value - exponent * 0.693145751953125;
    precise float product = exponent * 0.000001428606765330187;
    precise float error = fma(exponent, 0.000001428606765330187, -product);
    precise vec2 reduced = addPair(vec2(high, 0.0), vec2(-product, -error));
    precise vec2 polynomial = vec2(1.6059044372074283e-10, -5.3525266666590905e-18);
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(2.0876755879584152e-9, 1.1082839478332097e-16));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(2.5052107943679403e-8, 4.4176231769972645e-16));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(2.7557319981497130e-7, -7.5751123678698725e-15));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.0000027557318844628753, 3.7935713937038186e-14));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.000024801587642286904, -3.4069960259041840e-13));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.00019841270113829523, -2.7255968207233472e-12));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.0013888889225199819, -3.3631092919220174e-11));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.0083333337679505348, -4.3461720160287154e-10));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.041666667908430099, -1.2417634351336027e-9));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.16666667163372040, -4.9670537405344106e-9));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(0.50000000000000000, 0.0000000000000000));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(1.0000000000000000, 0.0000000000000000));
    polynomial = addPair(multiplyPair(polynomial, reduced.x), vec2(1.0000000000000000, 0.0000000000000000));
    polynomial = addPair(polynomial, multiplyPair(polynomial, reduced.y));
    precise float result = polynomial.x + polynomial.y;
    int power = int(exponent);
    if (power > 127) {
        return ldexp(result * 2.0, power - 1);
    }
    return ldexp(result, power);
}

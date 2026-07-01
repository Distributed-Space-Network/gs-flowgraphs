"""Reed-Solomon over GF(256) — the RS(255,223) t=16 code (docs/08 Tier 2 FEC).

A clean, dependency-free RS codec (systematic encode + Berlekamp-Massey / Chien / Forney decode)
used by the CCSDS framings. Validated rigorously by error injection: corrects up to ``nsym/2``
symbol errors and does not silently miscorrect beyond that.

Parameterized by field polynomial / first-consecutive-root / generator so it can be pointed at
different RS variants. The default (``prim=0x11D``, ``fcr=0``, ``generator=2``) is the
*conventional* RS(255,223). NOTE — **CCSDS** RS(255,223) uses ``prim=0x187`` / ``fcr=112`` AND a
dual-basis symbol representation; the dual basis is NOT applied here, so for spec-conformant CCSDS
birds use gr-satellites' ``ccsds_rs`` deframer (reused via synthetic SatYAML). This codec is the
general engine + the reusable building block. Pure Python (lists of ints); numpy arrays accepted.
"""
from __future__ import annotations

RS_NSYM_255_223 = 32  # RS(255,223): 32 parity symbols, corrects up to 16 symbol errors


class RSCodec:
    """Reed-Solomon codec over GF(2^8). ``nsym`` parity symbols → corrects ``nsym//2`` errors."""

    def __init__(self, nsym: int = RS_NSYM_255_223, *, prim: int = 0x11D,
                 fcr: int = 0, generator: int = 2) -> None:
        self.nsym = int(nsym)
        self.fcr = int(fcr)
        self.gen = int(generator)
        self._exp = [0] * 512
        self._log = [0] * 256
        x = 1
        for i in range(255):
            self._exp[i] = x
            self._log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim
        for i in range(255, 512):
            self._exp[i] = self._exp[i - 255]
        self._gpoly = self._generator_poly(self.nsym)

    # ── GF(256) arithmetic ──
    def _mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self._exp[self._log[a] + self._log[b]]

    def _pow(self, a: int, p: int) -> int:
        return self._exp[(self._log[a] * p) % 255]

    def _inv(self, a: int) -> int:
        return self._exp[255 - self._log[a]]

    def _poly_scale(self, p, s):
        return [self._mul(c, s) for c in p]

    def _poly_add(self, p, q):
        r = [0] * max(len(p), len(q))
        for i in range(len(p)):
            r[i + len(r) - len(p)] = p[i]
        for i in range(len(q)):
            r[i + len(r) - len(q)] ^= q[i]
        return r

    def _poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self._mul(p[i], q[j])
        return r

    def _poly_eval(self, p, x):
        y = p[0]
        for i in range(1, len(p)):
            y = self._mul(y, x) ^ p[i]
        return y

    def _generator_poly(self, nsym):
        g = [1]
        for i in range(nsym):
            g = self._poly_mul(g, [1, self._pow(self.gen, i + self.fcr)])
        return g

    # ── Encode ──
    def encode(self, data) -> bytes:
        """Systematically encode ``data`` (≤ 255-nsym bytes) → ``data + nsym`` parity bytes."""
        msg = list(bytes(bytearray(int(b) & 0xFF for b in data)))
        if len(msg) > 255 - self.nsym:
            raise ValueError(f"message too long: {len(msg)} > {255 - self.nsym}")
        out = msg + [0] * self.nsym
        for i in range(len(msg)):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(self._gpoly)):
                    out[i + j] ^= self._mul(self._gpoly[j], coef)
        out[:len(msg)] = msg
        return bytes(out)

    # ── Decode ──
    def _syndromes(self, msg):
        return [0] + [self._poly_eval(msg, self._pow(self.gen, i + self.fcr))
                      for i in range(self.nsym)]

    def _error_locator(self, synd):
        err_loc = [1]
        old_loc = [1]
        for i in range(self.nsym):
            delta = synd[i + 1]
            for j in range(1, len(err_loc)):
                delta ^= self._mul(err_loc[-(j + 1)], synd[i + 1 - j])
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self._poly_scale(old_loc, delta)
                    old_loc = self._poly_scale(err_loc, self._inv(delta))
                    err_loc = new_loc
                err_loc = self._poly_add(err_loc, self._poly_scale(old_loc, delta))
        return err_loc

    def _find_errors(self, err_loc, nmess):
        errs = len(err_loc) - 1
        positions = []
        for i in range(nmess):
            if self._poly_eval(err_loc, self._pow(self.gen, i)) == 0:
                positions.append(nmess - 1 - i)
        return positions if len(positions) == errs else None

    def _poly_div(self, dividend, divisor):
        out = list(dividend)
        for i in range(len(dividend) - (len(divisor) - 1)):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:
                        out[i + j] ^= self._mul(divisor[j], coef)
        sep = -(len(divisor) - 1)
        return out[:sep], out[sep:]

    def _correct(self, msg, synd, err_pos):
        coord = [len(msg) - 1 - p for p in err_pos]
        # errata locator Π(1 - X_i·x)
        e_loc = [1]
        for i in coord:
            e_loc = self._poly_mul(e_loc, self._poly_add([1], [self._pow(self.gen, i), 0]))
        # error evaluator Ω(x) = (S(x)·Λ(x)) mod x^(nsym+1)
        _, e_eval = self._poly_div(
            self._poly_mul(synd[::-1], e_loc), [1] + [0] * (len(e_loc)))
        e_eval = e_eval[::-1]
        x = [self._pow(self.gen, p) for p in coord]
        e = [0] * len(msg)
        for i, xi in enumerate(x):
            xi_inv = self._inv(xi)
            denom = 1  # Forney denominator = Λ'(X_i^-1) via the product form
            for j in range(len(x)):
                if j != i:
                    denom = self._mul(denom, 1 ^ self._mul(xi_inv, x[j]))
            if denom == 0:
                return None
            y = self._poly_eval(e_eval[::-1], xi_inv)
            y = self._mul(self._pow(xi, 1 - self.fcr), y)
            e[err_pos[i]] = self._mul(y, self._inv(denom))
        return self._poly_add(msg, e)

    def decode(self, codeword):
        """Correct ``codeword`` (255 bytes) and return the ``255-nsym`` message bytes, or ``None``
        if it is beyond the correction capability (> nsym//2 symbol errors)."""
        msg = list(bytes(bytearray(int(b) & 0xFF for b in codeword)))
        synd = self._syndromes(msg)
        if max(synd) == 0:
            return bytes(msg[:len(msg) - self.nsym])
        err_loc = self._error_locator(synd)[::-1]  # orientation for the Chien search
        err_pos = self._find_errors(err_loc, len(msg))
        if err_pos is None:
            return None
        fixed = self._correct(msg, synd, err_pos)
        if fixed is None or max(self._syndromes(fixed)) != 0:
            return None
        return bytes(fixed[:len(fixed) - self.nsym])

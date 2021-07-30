//
//  base64 with boost
//

#ifndef BASE64_BOOST_H_C0CE2A47_D10E_42C9_A27C_C883944E704A
#define BASE64_BOOST_H_C0CE2A47_D10E_42C9_A27C_C883944E704A

#include <string>
using namespace std;

bool Base64Encode(const string& input, string* output);
bool Base64Decode(const string& input, string* output);


#endif /* BASE64_BOOST_H_C0CE2A47_D10E_42C9_A27C_C883944E704A */
